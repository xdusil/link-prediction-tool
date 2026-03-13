
#include "LinkPredictionApp.hpp"
#include "Types.hpp"
#include "classifier/binary/BinaryRandomForestClassifier.hpp"
#include "classifier/generic/RandomForestClassifier.hpp"
#include "config/config.hpp"
#include "data_preprocessing/FlowProcessor.hpp"
#include "constrained_collections/reservoirs/CapacityLimitedReservoir.hpp"
#include "constrained_collections/counters/EvictingCounter.hpp"
#include "generators/context/SlidingWindowContextGenerator.hpp"
#include "exceptions/exceptions.hpp"
#include "generators/dependency/CandidateDependencyGenerator.hpp"
#include "generators/feature/FeatureGenerator.hpp"
#include "graph/boost/analytics/BoostGraphAnalytics.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "graph/network/NetworkGraphManager.hpp"
#include "io/FileReader.hpp"
#include "model/trainer/SkipGramTrainer.hpp"
#include "model/optimizer/Optimizer.hpp"
#include "model/data/DataLoader.hpp"
#include "random_walk/logic/custom/CustomRandomWalkLogic.hpp"
#include "random_walk/manager/RandomWalkManager.hpp"
#include "io/FileWriter.hpp"
#include "mlpack/core/data/scaler_methods/min_max_scaler.hpp"
#include "statistics/metrics.hpp"
#include "utils/ip/AllowedIPChecker.hpp"
#include "utils/ip/BoostIPHandler.hpp"
#include "utils/utils.hpp"
#include "utils/timers/timers.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <thread>

LinkPredictionApp::LinkPredictionApp(const std::optional<std::string> &config_path /*= std::nullopt*/,
                                     bool verbose /*= false*/) : m_verbose(verbose) {
    utils::VerboseTimer::set_verbose(verbose);
    log_verbose("Verbose mode enabled");
    log_verbose("Configuration path: ", config_path ? *config_path : "using defaults");

    // Load configuration
    if (config_path) {
        m_config = config::load(*config_path);
    } else {
        m_config = config::Config();
    }

    log_verbose("Loaded configuration: \n", "  - ", utils::to_string(m_config));

    // Initialize components
    m_internal_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_INTERNAL);
    m_external_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_EXTERNAL);
    m_reservoir =
        std::make_unique<CapacityLimitedReservoir<IPAddress, IPEdge>>(m_config.MAX_EDGES, m_seed);
    m_graph_manager = std::make_unique<NetworkGraphManager>();

    // Set threads
    m_config.NUM_THREADS = m_config.NUM_THREADS.value_or(std::thread::hardware_concurrency());
    if (*m_config.NUM_THREADS <= 0)
        m_config.NUM_THREADS = 1;

    utils::set_global_threads_count(*m_config.NUM_THREADS);
    log_verbose("Using ", *m_config.NUM_THREADS, " threads.");
}

void LinkPredictionApp::common_startup(std::optional<std::string> blocked_ips_path,
                                       std::optional<std::string> internal_ips_path) {
    // Initialize IP checkers
    m_allowed_ip_checker =
        std::make_unique<AllowedIPChecker>(std::nullopt, blocked_ips_path);
    m_internal_ip_checker = std::make_unique<BoostIPHandler>(internal_ips_path);

    m_dependency_analyzer = std::make_unique<ground_truth::DependencyAnalyzer>(
        m_config.N_OCCURRENCES, m_config.EPSILON, *m_allowed_ip_checker);
}

void LinkPredictionApp::common_training_or_prediction(
    std::string data_path, std::optional<std::string> blocked_ips_path,
    std::optional<std::string> internal_ips_path) {

    // Initialize common components
    common_startup(blocked_ips_path, internal_ips_path);

    // Process data and build graph
    process_data(data_path);
    build_graph();

    // Only generate embeddings if any embedding features are enabled
    if (m_config.FEATURE_CONFIG.are_embedding_features_enabled()) {
        generate_embeddings();
    } else {
        std::cout << "Skipping embedding generation (no embedding features enabled)" << std::endl;
        // Create a minimal model with 1 dimension to satisfy interface requirements
        m_model = std::make_unique<DirectionalSkipGramModel>(m_graph_manager->get_vertex_count(), 1);
    }
}

void LinkPredictionApp::run_ground_truth_mode(
    const std::string &data_path, const std::string &ground_truth_output_path,
    const std::optional<std::string> &blocked_ips_path) {
    std::cout << "Starting ground truth only mode..." << std::endl;
    utils::VerboseTimer timer("Ground truth calculation");

    // Initialize components
    common_startup(blocked_ips_path, std::nullopt);

    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");

    m_dependency_analyzer->calculate_dependencies(data_path, ground_truth_output_path);
    std::cout << "Ground truth calculation completed successfully." << std::endl;
}

void LinkPredictionApp::run_training_mode(
    const std::string &classifier_path, const std::string &data_path,
    const std::optional<std::string> &ground_truth_input_path /*= std::nullopt*/,
    const std::optional<std::string> &ground_truth_output_path /*= std::nullopt*/,
    const std::optional<std::string> &blocked_ips_path /*= std::nullopt*/,
    const std::optional<std::string> &internal_ips_path /*= std::nullopt*/,
    bool feature_importance /*= false*/) {

    std::cout << "Starting training mode..." << std::endl;
    utils::VerboseTimer timer("Training mode");

    // Initialize components
    common_training_or_prediction(data_path, blocked_ips_path, internal_ips_path);

    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");
    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");
    if (!m_model)
        throw ComponentNotInitializedException("Model not initialized.");

    if (ground_truth_input_path) {
        // Load ground truth
        m_dependency_analyzer->load_dependencies(*ground_truth_input_path);
    } else {
        // Calculate ground truth
        m_dependency_analyzer->calculate_dependencies(data_path,
                                                      ground_truth_output_path);
    }

    const auto &all_deps = m_dependency_analyzer->get_dependencies();
    DirectionalEmbeddings embeddings = m_model->get_embeddings();

    BoostGraphAnalytics analytics{*m_graph_manager};
    FeatureGenerator<BoostGraphTraits<Graph>, DirectionalEmbeddings,
                     decltype(all_deps)>
        feature_generator{analytics, embeddings, m_config.FEATURE_CONFIG};

    auto [combined, arma_labels] = feature_generator.generate_labeled_features(
        m_graph_manager->get_ip_to_vertex(), all_deps);

    // Convert to Armadillo matrix - no copy mem => ref needs to live
    auto [arma_features, ref] =
        utils::conv_2d_tensor_to_arma<float>(combined, false, true);

    std::cout << "Training data prepared.\n"
              << "  Features: " << arma_features.n_rows << "x" << arma_features.n_cols
              << "\n  Labels: " << arma_labels.n_elem << std::endl;
    
    log_verbose("Using features: ");
    log_verbose("  - ", utils::join(m_config.FEATURE_CONFIG.get_feature_names(), ", "));

    // Train the classifier
    train_classifier(arma_features, arma_labels, m_config.USE_GRID_SEARCH, feature_importance);

    if (!m_classifier)
        throw ComponentNotInitializedException("Classifier not initialized.");

    // Save the trained classifier
    m_classifier->save(classifier_path);
}

void LinkPredictionApp::run_prediction_mode(
    const std::string &classifier_path, const std::string &predictions_output_path,
    const std::string &data_path, const std::optional<std::string> &ground_truth_path,
    const std::optional<std::string> &blocked_ips_path,
    const std::optional<std::string> &internal_ips_path) {

    std::cout << "Starting prediction mode..." << std::endl;
    utils::VerboseTimer timer("Prediction mode");

    // Initialize components
    common_training_or_prediction(data_path, blocked_ips_path, internal_ips_path);

    // Load classifier
    m_classifier =
        std::make_unique<BinaryRandomForestClassifier<arma::fmat, arma::Row<std::size_t>>>();
    m_classifier->load(classifier_path);

    if (m_config.CLASSIFIER_THRESHOLD) {
        m_classifier->set_threshold(*m_config.CLASSIFIER_THRESHOLD);
    }

    // Generate predictions
    generate_predictions(predictions_output_path, ground_truth_path);

    std::cout << "Prediction completed successfully." << std::endl;
}

void LinkPredictionApp::generate_predictions(
    const std::string &output_path, const std::optional<std::string> &ground_truth_path) {
    std::cout << "Generating predictions..." << std::endl;

    if (!m_classifier)
        throw ComponentNotInitializedException("Classifier not initialized.");
    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");
    if (!m_model)
        throw ComponentNotInitializedException("Model not initialized.");
    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");

    BoostGraphAnalytics analytics{*m_graph_manager};
    FeatureGenerator<NetworkGraphManager::Base, decltype(m_model->get_embeddings()),
                     decltype(m_dependency_analyzer->get_dependencies())>
        feature_generator{analytics, m_model->get_embeddings(),
                          m_config.FEATURE_CONFIG};

    torch::Tensor combined;
    std::vector<std::pair<std::string, std::string>> vertex_pairs;
    std::optional<arma::Row<size_t>> labels = std::nullopt;

    if (ground_truth_path) {
        m_dependency_analyzer->load_dependencies(*ground_truth_path);
        auto [combined_val, labels_val, vertex_pairs_val] =
            feature_generator.generate_labeled_features_with_pairs(
                m_graph_manager->get_ip_to_vertex(),
                m_dependency_analyzer->get_dependencies());

        combined = std::move(combined_val);
        labels = std::move(labels_val);
        vertex_pairs = std::move(vertex_pairs_val);
    } else {
        auto [combined_val, vertex_pairs_val] =
            feature_generator.generate_unlabeled_features_with_pairs(
                m_graph_manager->get_ip_to_vertex());

        combined = std::move(combined_val);
        vertex_pairs = std::move(vertex_pairs_val);
    }

    // Convert to Armadillo matrix - no copy mem => ref needs to live
    auto [arma_features, ref] =
        utils::conv_2d_tensor_to_arma<float>(combined, false, true);

    const auto &predictions = m_classifier->predict(arma_features);

    // Write predictions to file
    FileWriter writer(output_path);
    writer.write_line("ip1,ip2");
    int positive_count = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto &pair = vertex_pairs[i];
        const auto &ip1 = pair.first;
        const auto &ip2 = pair.second;
        const auto &prediction = predictions[i];

        if (prediction == 1) {
            ++positive_count;
            writer.write_line(ip1 + "," + ip2);
        }
    }

    std::cout << "Positive predictions written to " << output_path << "\n";
    std::cout << "Positive predictions: " << positive_count << "\n";
    std::cout << "Total predictions: " << predictions.size() << std::endl;

    // Evaluate against ground truth if available
    if (labels.has_value()) {
        auto metrics = m_classifier->evaluate(arma_features, *labels);
        utils::print_classifier_metrics(metrics);
    }
}

void LinkPredictionApp::process_data(const std::string &data_path) {
    std::cout << "Processing data from " << data_path << std::endl;

    if (!m_internal_ip_checker)
        throw ComponentNotInitializedException("Internal IP checker not initialized.");
    if (!m_allowed_ip_checker)
        throw ComponentNotInitializedException("Allowed IP checker not initialized.");
    if (!m_internal_counter)
        throw ComponentNotInitializedException("Internal counter not initialized.");
    if (!m_external_counter)
        throw ComponentNotInitializedException("External counter not initialized.");
    if (!m_reservoir)
        throw ComponentNotInitializedException("Reservoir not initialized.");

    FlowProcessor processor(*m_internal_counter, *m_external_counter, *m_reservoir,
                            *m_allowed_ip_checker, *m_internal_ip_checker);

    // Process flow file
    processor.process_flow_file(data_path);      // fill internal and external counters
    processor.process_filtered_flows(data_path); // fill reservoir

    std::cout << "Processed flows: " << processor.get_total_flows_count() << "\n";
    std::cout << "Internal addresses: " << processor.get_internal_addresses_count() << "\n";
    std::cout << "External addresses: " << processor.get_external_addresses_count() << "\n";
    std::cout << "Total edges in reservoir: " << processor.get_total_edges_count() << std::endl;
    std::cout << "Data processing complete.\n";
}

void LinkPredictionApp::build_graph() {
    std::cout << "Building network graph..." << std::endl;

    if (!m_reservoir)
        throw ComponentNotInitializedException("Reservoir not initialized.");
    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");

    // Build graph from reservoir
    for (const auto &[ip, data] : *m_reservoir) {
        const auto &edges = data.first;
        for (const auto &edge : edges) {
            if (!m_graph_manager->add_edge_and_vertex_if_not_exists(
                    VertexProperties(edge.src_ip), VertexProperties(edge.dst_ip),
                    EdgeProperties(edge.start_timestamp, edge.end_timestamp,
                                   edge.protocol, edge.src_port.value_or(-1),
                                   edge.dst_port.value_or(-1)))) {
                std::cerr << "Could not add edge to the graph.\n";
            }
        }
    }

    std::cout << "Graph construction complete.\n";
    std::cout << "Vertices: " << m_graph_manager->get_vertex_count() << "\n";
    std::cout << "Edges: " << m_graph_manager->get_edge_count() << std::endl;

    if (m_graph_manager->get_vertex_count() == 0) {
        throw GraphEmptyException("Graph is empty.");
    }
}

void LinkPredictionApp::generate_embeddings() {
    std::cout << "Generating embeddings..." << std::endl;

    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");

    // Create random walk logic
    CustomRandomWalkLogic<NetworkGraphManager::Base, std::mt19937> walk_logic(
        m_config.N_APPEARANCES, m_config.EPSILON, m_config.EPSILON_REV);

    // Create random walk manager
    RandomWalkManager<NetworkGraphManager::Base> walk_manager(
        *m_graph_manager, m_config.NUM_THREADS.value(), walk_logic, 
        m_config.WALK_LENGTH, m_config.WALKS_PER_VERTEX, m_seed);

    // Set up context and dependency generators
    SlidingWindowContextGenerator<Vertex> context_generator(m_config.CONTEXT_SIZE);
    CandidateDependencyGenerator<Vertex> dependency_generator(
        m_graph_manager->get_vertex_count(), [](Vertex vertex) { return vertex; },
        m_config.NUM_NEGATIVE_SAMPLES, m_seed);

    // Create data loader with batching support
    DataLoader<Vertex> data_loader(walk_manager, context_generator, dependency_generator,
                                    m_config.BATCH_SIZE, m_verbose);


    // Create and train the Skip-Gram model
    m_model = std::make_unique<DirectionalSkipGramModel>(m_graph_manager->get_vertex_count(),
                                              m_config.EMBEDDING_DIM);

    // Train model
    Optimizer optimizer(*m_model, m_config.LEARNING_RATE);
    
    using BatchType = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
    GenericTrainer<DirectionalSkipGramModel, BatchType> trainer(
        *m_model, data_loader, optimizer, m_verbose);
    
    trainer.train(m_config.EPOCHS);

    std::cout << "Embedding generation complete." << std::endl;
}

void LinkPredictionApp::train_classifier(const auto &features, const auto &labels,
                                         bool use_grid_search, bool feature_importance) {

    std::cout << "Training classifier..." << std::endl;

    if (use_grid_search) {
        // Perform grid search
        RandomForestParams best_params = perform_grid_search(features, labels);
        m_classifier =
            std::make_unique<BinaryRandomForestClassifier<arma::fmat, arma::Row<std::size_t>>>(
                best_params, m_config.USE_SCALING);
    } else {
        // Create classifier
        m_classifier =
            std::make_unique<BinaryRandomForestClassifier<arma::fmat, arma::Row<std::size_t>>>(m_config.RF_PARAMS, m_config.USE_SCALING);
    }

    if (m_config.CLASSIFIER_THRESHOLD) {
        m_classifier->set_threshold(*m_config.CLASSIFIER_THRESHOLD);
    }

    // Train classifier
    if (m_config.USE_THRESHOLD_CALIBRATION) {
        m_classifier->train_with_calibration(features, labels, m_config.USE_WEIGHTS, m_config.METRIC_TO_OPTIMIZE);
    } else {
        m_classifier->train(features, labels, m_config.USE_WEIGHTS);
    }
    

    // Evaluate classifier
    auto metrics = m_classifier->evaluate(features, labels);
    utils::print_classifier_metrics(metrics);
    
    // Calculate feature importance
    if (feature_importance) {
        std::cout << "\n=== Feature Importance Analysis ===" << std::endl;
        auto feature_names = m_config.FEATURE_CONFIG.get_feature_names();
        auto importance = m_classifier->calculate_feature_importance(
            features, labels, feature_names, m_config.METRIC_TO_OPTIMIZE, 5);
        
        std::cout << "\nTop 20 Most Important Features:" << std::endl;
        for (std::size_t i = 0; i < std::min(std::size_t(20), importance.size()); ++i) {
            std::cout << "  " << (i + 1) << ". " << importance[i].first 
                     << ": " << importance[i].second << std::endl;
        }
        
        // Group analysis
        std::cout << "\nFeature Group Analysis:" << std::endl;
        double emb_importance = 0.0, topo_importance = 0.0, temporal_importance = 0.0,
               bidir_importance = 0.0, port_importance = 0.0;
        int emb_count = 0, topo_count = 0, temporal_count = 0, bidir_count = 0, port_count = 0;
        
        for (const auto &[name, score] : importance) {
            if (name.find("embedding") != std::string::npos || 
                name.find("cosine") != std::string::npos ||
                name.find("dot_product") != std::string::npos ||
                name.find("l1_distance") != std::string::npos ||
                name.find("l2_distance") != std::string::npos ||
                name.find("hadamard") != std::string::npos ||
                name.find("std") != std::string::npos ||
                name.find("abs_mean") != std::string::npos ||
                name.find("norm_ratio") != std::string::npos) {
                emb_importance += score;
                emb_count++;
            } else if (name.find("common_neighbors") != std::string::npos ||
                      name.find("jaccard") != std::string::npos ||
                      name.find("adamic") != std::string::npos ||
                      name.find("preferential") != std::string::npos ||
                      name.find("resource") != std::string::npos ||
                      name.find("degree") != std::string::npos) {
                topo_importance += score;
                topo_count++;
            } else if (name.find("temporal") != std::string::npos) {
                temporal_importance += score;
                temporal_count++;
            } else if (name.find("bidirectional") != std::string::npos) {
                bidir_importance += score;
                bidir_count++;
            } else if (name.find("port") != std::string::npos) {
                port_importance += score;
                port_count++;
            }
        }
        
        auto calc_avg = [](double total, int count) { return count > 0 ? total / count : 0.0; };
        
        std::cout << "  Embedding features:      total=" << emb_importance 
                  << " count=" << emb_count << " avg=" << calc_avg(emb_importance, emb_count) << "\n"
                  << "  Topology features:       total=" << topo_importance 
                  << " count=" << topo_count << " avg=" << calc_avg(topo_importance, topo_count) << "\n"
                  << "  Temporal features:       total=" << temporal_importance 
                  << " count=" << temporal_count << " avg=" << calc_avg(temporal_importance, temporal_count) << "\n"
                  << "  Bidirectional features:  total=" << bidir_importance 
                  << " count=" << bidir_count << " avg=" << calc_avg(bidir_importance, bidir_count) << "\n"
                  << "  Port features:           total=" << port_importance 
                  << " count=" << port_count << " avg=" << calc_avg(port_importance, port_count) << "\n"
                  << "\n=== End Feature Importance ===" << std::endl;
    }
}

RandomForestParams LinkPredictionApp::perform_grid_search(const auto &features,
                                                          const auto &labels) const {

    std::cout << "Performing grid search for hyperparameters...\n";

    const GridSearchParams &grid_params = m_config.GRID_PARAMS;
    std::cout << ">> numTrees: " << grid_params.num_trees << std::endl;
    std::cout << ">> minLeafSize: " << grid_params.min_leaf_size << std::endl;
    std::cout << ">> minGainSplit: " << grid_params.min_gain_split << std::endl;
    std::cout << ">> maxDepth: " << grid_params.max_depth << std::endl;
    std::cout << ">> validationSize: " << grid_params.validation_size << std::endl;

    // Perform grid search
    RandomForestParams best_params;
    double best_score;

    std::tie(best_params, best_score) =
        RandomForestClassifier<decltype(features), decltype(labels)>::
            grid_search(
                features, labels, 2, grid_params, m_config.USE_SCALING, m_config.USE_WEIGHTS,
                m_config.METRIC_TO_OPTIMIZE);

    std::cout << "Grid search complete." << std::endl;
    std::cout << "Best parameters:" << std::endl;
    std::cout << "  numTrees: " << best_params.num_trees << std::endl;
    std::cout << "  minLeafSize: " << best_params.min_leaf_size << std::endl;
    std::cout << "  maxDepth: " << best_params.max_depth << std::endl;
    std::cout << "  minGainSplit: " << best_params.min_gain_split << std::endl;
    std::cout << "  Best score: " << best_score << std::endl;

    return best_params;
}

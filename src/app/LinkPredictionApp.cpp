
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
#include "generators/embedding/EmbeddingTrainingPairGenerator.hpp"
#include "generators/feature/FeatureGenerator.hpp"
#include "graph/boost/analytics/BoostGraphAnalytics.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "graph/network/NetworkGraphManager.hpp"
#include "model/DirectionalEmbeddings.hpp"
#include "model/data/DataLoader.hpp"
#include "model/trainer/GenericTrainer.hpp"
#include "model/optimizer/Optimizer.hpp"
#include "model/DirectionalSkipGramModel.hpp"
#include "random_walk/logic/custom/CustomRandomWalkLogic.hpp"
#include "random_walk/manager/RandomWalkManager.hpp"
#include "reporting/PredictionReporter.hpp"
#include "service/EdgeServiceClassifier.hpp"
#include "service/ServicePortConfig.hpp"
#include "utils/ip/AllowedIPChecker.hpp"
#include "utils/ip/BoostIPHandler.hpp"
#include "utils/stream/OstreamFormatGuard.hpp"
#include "utils/utils.hpp"
#include "utils/timers/timers.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

LinkPredictionApp::LinkPredictionApp(const std::optional<std::string> &config_path /*= std::nullopt*/,
                                     bool verbose /*= false*/) : m_verbose(verbose) {
    utils::VerboseTimer::set_verbose(verbose);
    m_config_path = config_path;
    log_verbose("Verbose mode enabled");
    log_verbose("Configuration path: ", config_path ? *config_path : "using defaults");

    // Load configuration
    if (config_path) {
        m_config = config::load(*config_path);
    } else {
        m_config = config::Config();
    }

    log_verbose("Loaded configuration: \n", "  - ", utils::to_string(m_config));

    utils::set_global_seed(m_config.SEED);
    log_verbose("Using random seed: ", m_config.SEED);

    // Initialize components
    m_internal_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_INTERNAL);
    m_external_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_EXTERNAL);
    m_reservoir = std::make_unique<CapacityLimitedReservoir<IPAddress, IPEdge>>(
        m_config.MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET, m_config.SEED);
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
        m_model = std::make_unique<DirectionalSkipGramModel>(
            m_graph_manager->get_vertex_count(), 1);
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

    const ground_truth::DependencySet &ground_truth_dependencies =
        m_dependency_analyzer->get_dependencies();
    DirectionalEmbeddings embeddings = m_model->get_embeddings();

    BoostGraphAnalytics analytics{*m_graph_manager};
    if (m_config.WRITE_RUN_MANIFESTS) {
        reporting::write_run_manifest({
            .path = classifier_path + ".run_manifest.json",
            .mode = "training",
            .config_path = m_config_path,
            .data_path = data_path,
            .classifier_path = classifier_path,
            .output_path = classifier_path,
            .reference_path = reference_path,
            .config = m_config,
            .graph_manager = *m_graph_manager,
            .evaluated_pair_count = arma_labels.n_elem,
        });
    }

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
    const std::optional<std::string> &internal_ips_path,
    const std::optional<std::string> &scores_output_path) {

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
    const std::size_t evaluated_pair_count =
        generate_predictions(predictions_output_path, ground_truth_path,
                             scores_output_path);
    if (m_config.WRITE_RUN_MANIFESTS) {
        reporting::write_run_manifest({
            .path = predictions_output_path + ".run_manifest.json",
            .mode = "prediction",
            .config_path = m_config_path,
            .data_path = data_path,
            .classifier_path = classifier_path,
            .output_path = predictions_output_path,
            .reference_path = ground_truth_path,
            .config = m_config,
            .graph_manager = *m_graph_manager,
            .evaluated_pair_count = evaluated_pair_count,
        });
    }

    std::cout << "Prediction completed successfully." << std::endl;
}

std::size_t LinkPredictionApp::generate_predictions(
    const std::string &output_path,
    const std::optional<std::string> &ground_truth_path,
    const std::optional<std::string> &scores_output_path) {
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
    DirectionalEmbeddings embeddings = m_model->get_embeddings();
    FeatureGenerator<BoostGraphTraits<Graph>, DirectionalEmbeddings,
                     ground_truth::DependencySet>
        feature_generator{analytics, embeddings, m_config.FEATURE_CONFIG};

    torch::Tensor combined;
    std::optional<arma::Row<std::size_t>> labels = std::nullopt;
    std::vector<std::pair<IPAddress, IPAddress>> evaluated_pairs;

    if (ground_truth_path) {
        m_dependency_analyzer->load_dependencies(*ground_truth_path);
        auto [combined_val, labels_val, pairs_val] =
            feature_generator.generate_labeled_features_with_pairs(
                m_graph_manager->get_ip_to_vertex(),
                m_dependency_analyzer->get_dependencies());

        combined = std::move(combined_val);
        labels = std::move(labels_val);
        evaluated_pairs = std::move(pairs_val);
    } else {
        auto [combined_val, pairs_val] =
            feature_generator.generate_unlabeled_features_with_pairs(
                m_graph_manager->get_ip_to_vertex());
        combined = std::move(combined_val);
        evaluated_pairs = std::move(pairs_val);
    }

    // Convert to Armadillo matrix - no copy mem => ref needs to live
    auto [arma_features, ref] =
        utils::conv_2d_tensor_to_arma<float>(combined, false, true);

    const auto [predictions, probabilities] = m_classifier->predict_proba(arma_features);
    const arma::rowvec positive_scores = probabilities.row(1);
    if (scores_output_path) {
        reporting::write_pair_scores(*scores_output_path, evaluated_pairs, predictions,
                                     positive_scores, labels);
        std::cout << "All pair scores written to " << *scores_output_path << std::endl;
    }

    std::optional<service::ServicePortConfig> service_port_config;
    std::optional<service::EdgeServiceClassifier<BoostGraphTraits<Graph>>> service_classifier;
    
    if (m_config.SERVICE_CONFIG.enabled) {
        service_port_config.emplace();
        service_port_config->load(m_config.SERVICE_CONFIG.port_config_path);
        
        service::ServiceClassificationConfig svc_config;
        svc_config.ephemeral_port_min = m_config.SERVICE_CONFIG.ephemeral_port_min;
        svc_config.min_flows = m_config.SERVICE_CONFIG.min_flows;
        svc_config.min_confidence = m_config.SERVICE_CONFIG.min_confidence;
        svc_config.smoothing_alpha = m_config.SERVICE_CONFIG.smoothing_alpha;
        svc_config.top_k = m_config.SERVICE_CONFIG.top_k;
        
        service_classifier.emplace(*service_port_config, svc_config);
        std::cout << "Service classification enabled with " 
                  << service_port_config->port_count() << " port mappings" << std::endl;
    }

    const service::EdgeServiceClassifier<BoostGraphTraits<Graph>>* service_classifier_ptr =
        service_classifier.has_value() ? &service_classifier.value() : nullptr;
    const reporting::PredictionWriteSummary prediction_summary =
        reporting::write_positive_predictions(output_path, evaluated_pairs, predictions,
                                              positive_scores, *m_graph_manager,
                                              service_classifier_ptr);

    std::cout << "Positive predictions written to " << output_path << "\n";
    std::cout << "Positive predictions: "
              << prediction_summary.positive_predictions << "\n";
    std::cout << "Total predictions: " << prediction_summary.total_predictions
              << std::endl;

    // Evaluate against ground truth if available
    if (labels.has_value()) {
        auto metrics = m_classifier->evaluate(arma_features, *labels);
        utils::print_classifier_metrics(metrics);
        const auto metrics_path = output_path + ".metrics.json";
        reporting::write_metrics_report(
            metrics_path, metrics,
            prediction_summary.positive_predictions,
            prediction_summary.total_predictions);
        std::cout << "Metrics report written to " << metrics_path << std::endl;
        const auto [optimal_threshold, optimal_metrics] =
            m_classifier->find_optimal_threshold(
                arma_features, *labels, m_config.METRIC_TO_OPTIMIZE);
        std::cout << "Optimal threshold: " << optimal_threshold << std::endl;
        utils::print_classifier_metrics(optimal_metrics);
    }

    return evaluated_pairs.size();
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
                            *m_allowed_ip_checker, *m_internal_ip_checker,
                            static_cast<std::size_t>(m_config.TEMPORAL_BUCKETS));

    std::cout << "Structured sampling:\n"
              << "  Endpoint retention: top internal/external endpoints by exact score\n"
              << "  Temporal buckets: " << m_config.TEMPORAL_BUCKETS << "\n"
              << "  Max edges per directed pair temporal bucket: "
              << m_config.MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET
              << std::endl;

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

    auto reservoir_keys = m_reservoir->get_keys();
    std::sort(reservoir_keys.begin(), reservoir_keys.end());

    for (const auto &key : reservoir_keys) {
        const std::vector<IPEdge> &edges = m_reservoir->get(key);
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
        m_config.WALK_LENGTH, m_config.WALKS_PER_VERTEX, m_config.SEED);

    // Set up context and embedding training-pair generators
    SlidingWindowContextGenerator<Vertex> context_generator(m_config.CONTEXT_SIZE);
    EmbeddingTrainingPairGenerator<Vertex> pair_generator(
        m_graph_manager->get_vertex_count(), [](Vertex vertex) { return vertex; },
        m_config.NUM_NEGATIVE_SAMPLES, m_config.SEED);

    // Create data loader with batching support
    DataLoader<Vertex> data_loader(walk_manager, context_generator, pair_generator,
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
            std::make_unique<
                BinaryRandomForestClassifier<arma::fmat, arma::Row<std::size_t>>>(
                m_config.RF_PARAMS, m_config.USE_SCALING);
    }

    if (m_config.CLASSIFIER_THRESHOLD) {
        m_classifier->set_threshold(*m_config.CLASSIFIER_THRESHOLD);
    }

    // Train classifier
    if (m_config.USE_THRESHOLD_CALIBRATION) {
        m_classifier->train_with_calibration(
            features, labels, m_config.USE_WEIGHTS, m_config.METRIC_TO_OPTIMIZE);
    } else {
        m_classifier->train(features, labels, m_config.USE_WEIGHTS);
    }
    

    // Evaluate classifier
    auto metrics = m_classifier->evaluate(features, labels);
    utils::print_classifier_metrics(metrics);
    
    // Calculate feature importance
    if (feature_importance) {
        const utils::OstreamFormatGuard format_guard(std::cout);
        std::cout << "\n=== Feature Importance Analysis ===" << std::endl;
        std::vector<std::string> feature_names = m_config.FEATURE_CONFIG.get_feature_names();
        auto importance = m_classifier->calculate_feature_importance(
            features, labels, feature_names, m_config.METRIC_TO_OPTIMIZE, 5);
        
        std::cout << "\nTop 20 Most Important Features:" << std::endl;
        for (std::size_t i = 0; i < std::min(std::size_t(20), importance.size()); ++i) {
            std::cout << "  " << (i + 1) << ". " << importance[i].first 
                     << ": " << importance[i].second << std::endl;
        }
        
        // Group analysis
        std::cout << "\nFeature Group Analysis:" << std::endl;
        double uncategorized_importance = 0.0;
        int uncategorized_count = 0;
        std::vector<std::string> uncategorized_features;

        struct GroupSummary {
            const char *name;
            const char *prefix;
            double total;
            int count;
        };

        std::array<GroupSummary, 5> groups = {{
            {"Embedding", "emb_", 0.0, 0},
            {"Topology", "struct_", 0.0, 0},
            {"Temporal", "time_", 0.0, 0},
            {"Observed Flow", "flow_", 0.0, 0},
            {"Network", "net_", 0.0, 0}
        }};

        auto has_prefix = [](const std::string &name, const char *prefix) {
            return name.rfind(prefix, 0) == 0;
        };
        
        for (const auto &[name, score] : importance) {
            bool matched = false;
            for (auto &group : groups) {
                if (has_prefix(name, group.prefix)) {
                    group.total += score;
                    ++group.count;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                uncategorized_importance += score;
                uncategorized_count++;
                uncategorized_features.push_back(name);
            }
        }
        
        auto calc_avg = [](double total, int count) { return count > 0 ? total / count : 0.0; };
        double grand_total = 0.0;
        int total_features = 0;
        for (const auto &group : groups) {
            grand_total += group.total;
            total_features += group.count;
        }

        auto calc_share = [](double total, double overall) {
            return overall > 0.0 ? (100.0 * total / overall) : 0.0;
        };

        std::cout << "\n"
                  << "  " << std::left << std::setw(15) << "Group"
                  << std::right << std::setw(8) << "Count"
                  << std::setw(14) << "Total"
                  << std::setw(14) << "Avg"
                  << std::setw(12) << "Share(%)" << "\n"
                  << "  " << std::string(63, '-') << "\n";

        std::cout << std::fixed << std::setprecision(6);
        for (const auto& group : groups) {
            std::cout << "  " << std::left << std::setw(15) << group.name
                      << std::right << std::setw(8) << group.count
                      << std::setw(14) << group.total
                      << std::setw(14) << calc_avg(group.total, group.count)
                      << std::setw(12) << calc_share(group.total, grand_total)
                      << "\n";
        }

        std::cout << "  " << std::string(63, '-') << "\n"
                  << "  " << std::left << std::setw(15) << "Total"
                  << std::right << std::setw(8) << total_features
                  << std::setw(14) << grand_total
                  << std::setw(14) << calc_avg(grand_total, total_features)
                  << std::setw(12) << 100.0 << "\n"
                  << "\n=== End Feature Importance ===" << std::endl;

        if (uncategorized_count > 0) {
            std::cout << "WARNING: " << uncategorized_count
                      << " feature(s) could not be assigned to any known group prefix."
                      << " Uncategorized total importance=" << uncategorized_importance
                      << "\n  Uncategorized features: "
                      << utils::join(uncategorized_features, ", ") << std::endl;
        }
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

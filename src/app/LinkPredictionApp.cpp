
#include "LinkPredictionApp.hpp"
#include "Types.hpp"
#include "classifier/RandomForestClassifier.hpp"
#include "config/config.hpp"
#include "exceptions/exceptions.hpp"
#include "generators/embedding/EmbeddingGenerator.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "io/FileReader.hpp"
#include "io/FileWriter.hpp"
#include "mlpack/core/data/scaler_methods/min_max_scaler.hpp"
#include "statistics/metrics.hpp"
#include "utils/ip/AllowedIPChecker.hpp"
#include "utils/ip/BoostIPHandler.hpp"
#include "utils/utils.hpp"
#include <fstream>
#include <iostream>
#include <optional>

LinkPredictionApp::LinkPredictionApp(const std::optional<std::string> &config_path) {
    // Load configuration
    if (config_path) {
        m_config = config::load(*config_path);
    } else {
        m_config = config::Config();
    }

    // Initialize components
    m_internal_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_INTERNAL);
    m_external_counter =
        std::make_unique<EvictingCounter<IPAddress>>(m_config.COUNT_EXTERNAL);
    m_reservoir =
        std::make_unique<CapacityLimitedReservoir<IPAddress, IPEdge>>(m_config.MAX_EDGES);
    m_graph_manager = std::make_unique<NetworkGraphManager>();

    // set seed
    m_seed = std::chrono::system_clock::now().time_since_epoch().count();
}

void LinkPredictionApp::common_training_or_prediction(
    std::string classifier_path, std::string data_path,
    std::optional<std::string> blocked_ips_path,
    std::optional<std::string> internal_ips_path) {

    // Initialize IP checkers
    m_allowed_ip_checker =
        std::make_unique<AllowedIPChecker>(std::nullopt, blocked_ips_path);
    m_internal_ip_checker = std::make_unique<BoostIPHandler>(internal_ips_path);
    m_dependency_analyzer = std::make_unique<ground_truth::DependencyAnalyzer>(
        m_config.N_OCCURRENCES, m_config.EPSILON, *m_allowed_ip_checker);

    // Process data and build graph
    process_data(data_path);
    build_graph();

    // Generate embeddings
    generate_embeddings();
}

void LinkPredictionApp::run_training_mode(
    const std::string &classifier_path, const std::string &data_path,
    const std::optional<std::string> &ground_truth_input_path /*= std::nullopt*/,
    const std::optional<std::string> &ground_truth_output_path /*= std::nullopt*/,
    const std::optional<std::string> &blocked_ips_path /*= std::nullopt*/,
    const std::optional<std::string> &internal_ips_path /*= std::nullopt*/,
    bool use_grid_search /*= false*/) {

    std::cout << "Starting training mode..." << std::endl;

    // Initialize components
    common_training_or_prediction(classifier_path, data_path, blocked_ips_path,
                                  internal_ips_path);

    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");
    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");

    if (ground_truth_input_path) {
        // Load ground truth
        m_dependency_analyzer->load_dependencies(*ground_truth_input_path);
    } else {
        // Calculate ground truth
        m_dependency_analyzer->calculate_dependencies(data_path,
                                                      ground_truth_output_path);
    }

    const auto &all_deps = m_dependency_analyzer->get_dependencies();

    EmbeddingGenerator<Vertex, decltype(m_model->get_embeddings()), decltype(all_deps)>
        embedding_generator{};

    // Generate embeddings and labels
    auto [combined, arma_labels] =
        embedding_generator.generate_dependency_embeddings_and_labels(
            m_graph_manager->get_ip_to_vertex(), all_deps, m_model->get_embeddings());

    // Convert to Armadillo matrix - no copy mem => ref needs to live
    auto [arma_features, ref] =
        utils::conv_2d_tensor_to_arma<float>(combined, false, true);

    std::cout << "Training data prepared.\n"
              << "  Features: " << arma_features.n_rows << "x" << arma_features.n_cols
              << "\n  Labels: " << arma_labels.n_elem << std::endl;

    // Train the classifier
    train_classifier(arma_features, arma_labels, use_grid_search);

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

    // Initialize components
    common_training_or_prediction(classifier_path, data_path, blocked_ips_path,
                                  internal_ips_path);

    // Load classifier
    m_classifier =
        std::make_unique<RandomForestClassifier<arma::fmat, arma::Row<size_t>>>();
    m_classifier->load(classifier_path);

    // Generate predictions
    generate_predictions(predictions_output_path, ground_truth_path);

    std::cout << "Prediction completed successfully." << std::endl;
}

void LinkPredictionApp::generate_predictions(const std::string &output_path, const std::optional<std::string> &ground_truth_path) {
    std::cout << "Generating predictions..." << std::endl;

    if (!m_classifier)
        throw ComponentNotInitializedException("Classifier not initialized.");
    if (!m_graph_manager)
        throw ComponentNotInitializedException("Graph manager not initialized.");
    if (!m_model)
        throw ComponentNotInitializedException("Model not initialized.");
    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");

    EmbeddingGenerator<
        Vertex, decltype(m_model->get_embeddings()),
        decltype(m_dependency_analyzer->get_dependencies())>
        embedding_generator{};

    auto [combined, vertex_pairs] =
        embedding_generator.generate_dependency_embeddings_and_vertex_pairs(
            m_graph_manager->get_ip_to_vertex(), m_model->get_embeddings());

    // Convert to Armadillo matrix - no copy mem => ref needs to live
    auto [arma_features, ref] =
        utils::conv_2d_tensor_to_arma<float>(combined, false, true);

    const auto &predictions = m_classifier->predict(arma_features);

    // Write predictions to file
    FileWriter writer(output_path);
    writer.write_line("src_ip,dst_ip");
    int positive_count = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto &pair = vertex_pairs[i];
        const auto &src_ip = pair.first;
        const auto &dst_ip = pair.second;
        const auto &prediction = predictions[i];

        if (prediction == 1) {
            ++positive_count;
        }
        writer.write_line(src_ip + "," + dst_ip);
    }

    std::cout << "Predictions written to " << output_path << "\n";
    std::cout << "Positive predictions: " << positive_count << "\n";
    std::cout << "Total predictions: " << predictions.size() << std::endl;

    if (ground_truth_path) {
        evaluate_predictions(predictions, vertex_pairs, *ground_truth_path);
    }
    evaluate_predictions(predictions, vertex_pairs, "ground_truth.txt");
}


void LinkPredictionApp::evaluate_predictions(const auto &predictions, const auto &vertex_pairs, const std::string &ground_truth_path) {
    std::cout << "Evaluating predictions..." << std::endl;

    if (!m_dependency_analyzer)
        throw ComponentNotInitializedException("Dependency analyzer not initialized.");
    if (!m_classifier)
        throw ComponentNotInitializedException("Classifier not initialized.");
    if (vertex_pairs.size() != predictions.size())
        throw std::runtime_error("Mismatch between predictions and vertex pairs.");

    m_dependency_analyzer->load_dependencies(ground_truth_path);
    const auto &all_deps = m_dependency_analyzer->get_dependencies();

    arma::Row<size_t> arma_labels(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto &pair = vertex_pairs[i];
        const auto &src_ip = pair.first;
        const auto &dst_ip = pair.second;
        arma_labels[i] =
            all_deps.contains({src_ip, dst_ip}) || all_deps.contains({dst_ip, src_ip})
                ? 1
                : 0;
    }

    //auto metrics = m_classifier->evaluate(arma_features, arma_labels);
    statistics::Metrics metrics{};
    std::cout << "Classifier evaluation metrics:\n"
              << "  Accuracy: " << metrics.accuracy << "\n"
              << "  Precision: " << metrics.precision << "\n"
              << "  Recall: " << metrics.recall << "\n"
              << "  F1 Score: " << metrics.f1_score << std::endl;

}
// Evaluate a classifier using a train/test split.
template <typename Features, typename Labels>
void evaluate_model_train_test_split(const RandomForestParams &params,
                                     const Features &features, const Labels &labels,
                                     double test_size = 0.25, bool use_scaling = true) {
    arma::mat features_d = arma::conv_to<arma::mat>::from(features);

    size_t num_classes = 2;
    RandomForestClassifier<arma::mat, Labels> rf(num_classes, params, use_scaling);

    // Containers for the split data.
    arma::mat train_features, test_features;
    Labels train_labels, test_labels;

    // Split the data. (test_size indicates the fraction of columns for testing.)
    // mlpack::data::StratifiedSplit(features_d, labels, trainFeatures, testFeatures,
    //                               trainLabels, testLabels, test_size);

    mlpack::data::Split(features_d, labels, train_features, test_features, train_labels,
                        test_labels, test_size);

    std::cout << "Train test split evaluation results with test size = " << test_size
              << ":\n";

    std::cout << "Train labels: \n";
    std::cout << "---> Positive labels: " << arma::accu(train_labels) << std::endl;
    std::cout << "---> Negative labels: "
              << train_labels.size() - arma::accu(train_labels) << std::endl;

    std::cout << "\n\n";
    std::cout << "Test labels: \n";
    std::cout << "---> Positive labels: " << arma::accu(test_labels) << std::endl;
    std::cout << "---> Negative labels: " << test_labels.size() - arma::accu(test_labels)
              << std::endl;
    std::cout << "\n\n";

    rf.train(train_features, train_labels);
    Labels y_predicted = rf.predict(test_features);

    std::cout << "Predicted labels: \n";
    std::cout << "---> Positive labels: " << arma::accu(y_predicted) << std::endl;
    std::cout << "---> Negative labels: " << y_predicted.size() - arma::accu(y_predicted)
              << std::endl;
    std::cout << "\n\n";

    auto metrics = rf.evaluate(test_features, test_labels);
    std::cout << "Accuracy: " << metrics.accuracy << std::endl;
    std::cout << "Precision: " << metrics.precision << std::endl;
    std::cout << "Recall: " << metrics.recall << std::endl;
    std::cout << "F1 Score: " << metrics.f1_score << std::endl;
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

    // Count total edges
    int total_edges = 0;
    for (const auto &key : m_reservoir->get_keys()) {
        total_edges += m_reservoir->get_size(key);
    }

    std::cout << "Data processing complete.\n";
    std::cout << "Internal addresses: " << m_internal_counter->get_items().size() << "\n";
    std::cout << "External addresses: " << m_external_counter->get_items().size() << "\n";
    std::cout << "Total edges in reservoir: " << total_edges << std::endl;
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
        *m_graph_manager, m_config.NUM_THREADS, walk_logic, m_config.WALK_LENGTH, m_seed);

    // Set up context and dependency generators
    SlidingWindowContextGenerator<Vertex> context_generator(m_config.CONTEXT_SIZE);
    CandidateDependencyGenerator<Vertex> dependency_generator(
        [](const std::vector<Vertex> &context) { return context[0]; },
        m_graph_manager->get_vertex_count(), [](Vertex vertex) { return vertex; },
        m_config.NUM_NEGATIVE_SAMPLES);

    // Create data loader
    DataLoader<Vertex> data_loader(walk_manager, context_generator, dependency_generator);

    // Create and train the Skip-Gram model
    m_model = std::make_unique<SkipGramModel>(m_graph_manager->get_vertex_count(),
                                              m_config.EMBEDDING_DIM);

    // Train model
    Optimizer optimizer(*m_model, m_config.LEARNING_RATE);
    SkipGramTrainer trainer(*m_model, data_loader, optimizer);
    trainer.train(m_config.EPOCHS);

    std::cout << "Embedding generation complete." << std::endl;
}

void LinkPredictionApp::train_classifier(const auto &features, const auto &labels,
                                         bool use_grid_search) {

    std::cout << "Training classifier..." << std::endl;

    if (use_grid_search) {
        // Perform grid search
        RandomForestParams best_params = perform_grid_search(features, labels);
        m_classifier =
            std::make_unique<RandomForestClassifier<arma::fmat, arma::Row<size_t>>>(
                2, best_params.num_trees, best_params.min_leaf_size,
                best_params.min_gain_split, best_params.max_depth);
    } else {
        // Create classifier
        m_classifier =
            std::make_unique<RandomForestClassifier<arma::fmat, arma::Row<size_t>>>(
                2, m_config.NUM_TREES, m_config.MIN_LEAF_SIZE, m_config.MIN_GAIN_SPLIT,
                m_config.MAX_DEPTH);
    }

    // Train classifier
    m_classifier->train(features, labels);

    // Evaluate classifier
    statistics::Metrics metrics = m_classifier->evaluate(features, labels);
    std::cout << "Classifier training complete." << std::endl;
    std::cout << "Training metrics:" << std::endl;
    std::cout << "  Accuracy: " << metrics.accuracy << std::endl;
    std::cout << "  Precision: " << metrics.precision << std::endl;
    std::cout << "  Recall: " << metrics.recall << std::endl;
    std::cout << "  F1 Score: " << metrics.f1_score << std::endl;

    RandomForestParams params;
    params.num_trees = m_config.NUM_TREES;
    params.min_leaf_size = m_config.MIN_LEAF_SIZE;
    params.min_gain_split = m_config.MIN_GAIN_SPLIT;
    params.max_depth = m_config.MAX_DEPTH;

    ///////////
    evaluate_model_train_test_split(params, features, labels, 0.25);
    evaluate_model_train_test_split(params, features, labels, 0.5);
    evaluate_model_train_test_split(params, features, labels, 0.75);
    ///////////
}

RandomForestParams LinkPredictionApp::perform_grid_search(const auto &features,
                                                          const auto &labels) {

    std::cout << "Performing grid search for hyperparameters...\n";

    // Define hyperparameter ranges
    std::vector<std::size_t> num_trees = {10, 20, 50, 100};
    std::vector<std::size_t> min_leaf_size = {1, 3, 5};
    std::vector<double> min_gain_split = {0.0, 1e-7, 1e-5};
    std::vector<std::size_t> max_depth = {0, 10, 20, 30};

    std::cout << ">> numTrees: " << num_trees << std::endl;
    std::cout << ">> minLeafSize: " << min_leaf_size << std::endl;
    std::cout << ">> minGainSplit: " << min_gain_split << std::endl;
    std::cout << ">> maxDepth: " << max_depth << std::endl;

    // Perform grid search
    RandomForestParams best_params;
    double best_score;

    std::tie(best_params, best_score) =
        RandomForestClassifier<decltype(features), decltype(labels)>::
            template grid_search<mlpack::F1<mlpack::AverageStrategy::Binary>>(
                features, labels, 2, num_trees, min_leaf_size, min_gain_split, max_depth,
                0.3);

    std::cout << "Grid search complete." << std::endl;
    std::cout << "Best parameters:" << std::endl;
    std::cout << "  numTrees: " << best_params.num_trees << std::endl;
    std::cout << "  minLeafSize: " << best_params.min_leaf_size << std::endl;
    std::cout << "  maxDepth: " << best_params.max_depth << std::endl;
    std::cout << "  minGainSplit: " << best_params.min_gain_split << std::endl;
    std::cout << "  Best score: " << best_score << std::endl;

    return best_params;
}

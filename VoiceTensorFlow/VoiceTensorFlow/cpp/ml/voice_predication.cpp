

#include "voice_predication.hpp"

using namespace tensorflow;

VoicePrediction::VoicePrediction() {
	
}

bool VoicePrediction::loadGraphFromPath(const std::string& path) {

	auto status = ReadBinaryProto(tensorflow::Env::Default(), path, &graph);

	if (!status.ok()) {
		LOG(INFO) << "Error reading graph: " << status.error_message().c_str();
		return false;
	}
	
	// This prints out the names of the nodes in the graph.
	auto nodeCount = graph.node_size();
	LOG(INFO) << "Node count: " << nodeCount;

	for (auto i = 0; i < nodeCount; ++i) {
		auto node = graph.node(i);
		LOG(INFO) << "Node " << i << " " << node.op().c_str() << " " << node.name().c_str();
	}
	
	return true;
}

bool VoicePrediction::createSession() {
	
	SessionOptions options;
	auto status = tensorflow::NewSession(options, &session);

	if (!status.ok()) {
		LOG(INFO) << "Error creating session: " << status.error_message().c_str();
		return false;
	}
	
	status = session->Create(graph);
	if (!status.ok()) {
		LOG(INFO) << "Error adding graph to session: " << status.error_message().c_str();
		return false;
	}
	
	return true;
}

bool VoicePrediction::closeSession() {
	
	if (nullptr == this->session) {
		LOG(INFO) << "No instance session.";
		return false;
	}
	
	auto status = session->Close();
	
	if (!status.ok()) {
		LOG(INFO) << "Close session error.";
		return false;
	}
	
	LOG(INFO) << "Close session successfully";
	
	return true;
}


void VoicePrediction::predict(float *example) {
	// Define the tensor for the input data. This tensor takes one example
	// at a time, and the example has 20 features.
	tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 20 }));
	
	// Put the input data into the tensor.
	auto input = x.tensor<float, 2>();
	for (int i = 0; i < 20; ++i) {
		input(0, i) = example[i];
	}
	
	// The feed dictionary for doing inference.
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{"inputs/x-input", x}
	};
	
	// We want to run these nodes.
	std::vector<std::string> nodes = {
		{"model/y_pred"},
		{"inference/inference"}
	};
	
	// The results of running the nodes are stored in this vector.
	std::vector<tensorflow::Tensor> outputs;
	
	// Run the session.
	auto status = session->Run(inputs, nodes, {}, &outputs);
	if (!status.ok()) {
		LOG(INFO) << "Error running model: " << status.error_message().c_str();
		return;
	}
	
	// Print out the result of the prediction.
	auto y_pred = outputs[0].tensor<float, 2>();
	LOG(INFO) << "Probability spoken by a male: " << y_pred(0, 0);
	
	auto isMale = outputs[1].tensor<float, 2>();
	if (isMale(0, 0)) {
		LOG(INFO) << "Prediction: male";
	} else {
		LOG(INFO) << "Prediction: female";
	}
	
	
}





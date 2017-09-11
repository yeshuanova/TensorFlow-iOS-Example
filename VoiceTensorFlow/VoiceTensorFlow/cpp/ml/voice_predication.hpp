

#ifndef voice_predication_hpp
#define voice_predication_hpp

#include <stdio.h>

#include <string>
#include <tensorflow/core/public/session.h>


class VoicePrediction {
	
private:
	tensorflow::GraphDef graph;
	tensorflow::Session *session;
	
public:
	
	VoicePrediction();
	
	bool loadGraphFromPath(const std::string& path);
	
	bool createSession();
	
	bool closeSession();
	
	void predict(float *example);

};


#endif /* voice_predication_hpp */

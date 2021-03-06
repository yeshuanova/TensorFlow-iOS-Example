
#import "ViewController.h"

#include "voice_predication.hpp"

// I took these examples from the test set. To make this an app that is useful
// in practice, you would need to add code that records audio and then extracts
// the acoustic properties from the audio.
static float maleExample[] = {
	0.174272105181833,0.0694110453235828,0.190874106652007,0.115601979109401,
	0.228279274326553,0.112677295217152,4.48503835015822,61.7649083141473,
	0.950972024663856,0.635199178449614,0.0500274876305662,0.174272105181833,
	0.102045991451092,0.0183276059564719,0.246153846153846,1.62129934210526,
	0.0078125,7,6.9921875,0.209310986964618,
};

static float femaleExample[] = {
	0.19753980448103,0.0347746034366121,0.198683834048641,0.182660944206009,
	0.218712446351931,0.0360515021459227,2.45939730314823,9.56744874023233,
	0.839523285188244,0.226976814502006,0.185064377682403,0.19753980448103,
	0.173636160583901,0.0470127326150832,0.271186440677966,1.61474609375,
	0.2109375,15.234375,15.0234375,0.0389615584623385,
};

@implementation ViewController
{
	VoicePrediction vp;
}

- (void)viewDidLoad {
	[super viewDidLoad];

	NSString *path = [[NSBundle mainBundle] pathForResource:@"inference" ofType:@"pb"];

	if (vp.loadGraphFromPath(path.fileSystemRepresentation) && vp.createSession()) {
		vp.predict(maleExample);
		vp.predict(femaleExample);
		
		if (vp.closeSession()) {
			NSLog(@"Close session ok");
		}
	}

}

@end

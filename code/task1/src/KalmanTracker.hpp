#include <stdio.h>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Maximun number of char in the blob's format
const int MAX_FORMAT = 1024;

struct cvBlob {
	int     ID;  /* blob ID        */
	int   x, y;  /* blob position  */
	int   w, h;  /* blob sizes     */	
    int area;
	char format[MAX_FORMAT];
};

inline cvBlob initBlob(int id, int x, int y, int w, int h,int area)
{
	cvBlob B = { id,x,y,w,h,area};
	return B;
}

inline cvBlob initBlob(int id, Rect rectangle,int area)
{
	cvBlob B = { id,rectangle.x,rectangle.y,rectangle.width,rectangle.height,area};    
	return B;
}


//Allows to track the bigger detected foreground object for every frame given
class KalmanTracker{
    private:
        //Foreground detection variables
        double _learningrate;
        int _history;
        int _varThreshold;
        cv::Ptr<cv::BackgroundSubtractorMOG2> _pMOG2;

        //Blob extraction varibales        
        int _connectivity;
        int _min_width;
        int _min_height;

        KalmanFilter _KF;
        int _mode;
        float _xcenter;
        float _ycenter;

        bool _first_measurement;
        bool _first_frame;

        //blob extraction functions
        vector<cvBlob> _extractBlobs();
        void _makeMeasurement(Mat frame);
        void _initKalman();


    public:
        //Background subtraction images
        Mat frameGray;
        Mat fgmask;
        Mat fgmaskFiltered;

        //Kalman filter variables
        Mat state;
        Mat measurement;
        Mat prediction;

		Point measPt;
        Point statePt;
		Point predPt;
        
        //Constructor
        KalmanTracker(int mode);

        void track(Mat frame);
};
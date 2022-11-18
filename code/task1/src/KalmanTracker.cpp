                                                                                                                                                                                                                                                                                                                                           #include "KalmanTracker.hpp"

//Constructor
 KalmanTracker::KalmanTracker(int mode) {
    _learningrate = 0.001;
    _history = 50;
    _varThreshold = 16;
    _pMOG2 = cv::createBackgroundSubtractorMOG2();
	//_pMOG2->setDetectShadows(false);

    _pMOG2->setVarThreshold(_varThreshold);
	_pMOG2->setHistory(_history);

    _connectivity = 4;
    _min_width = 10;
    _min_height = 10;
    
    _mode = mode;
    _initKalman();

    _first_measurement = false;
    _first_frame = false;

	if(_mode==0){
		_xcenter = 0;
		_ycenter = 2;
	}
	else{
		_xcenter = 0;
		_ycenter = 3;
	}
 }

void KalmanTracker::track(Mat frame){

	if(_first_measurement){
		prediction = _KF.predict();
		prediction.copyTo(state);
	}

    _makeMeasurement(frame);
    
    if(measurement.at<float>(0)>0){
		if(!_first_measurement){
			_KF.statePost.at<float>(_xcenter) = measurement.at<float>(0);
			_KF.statePost.at<float>(_ycenter) = measurement.at<float>(1);
			state.at<float>(_xcenter) = measurement.at<float>(0);
			state.at<float>(_ycenter) = measurement.at<float>(1);
			_first_measurement = true;
		}
		else{state = _KF.correct(measurement);}
	}

	statePt.x = state.at<float>(_xcenter);
	statePt.y = state.at<float>(_ycenter);	
	predPt.x = prediction.at<float>(_xcenter);
	predPt.y = prediction.at<float>(_ycenter);
}

vector<cvBlob> KalmanTracker::_extractBlobs()
{																											
	Mat aux; 																								// auxiliary variables
	int area; 
	cvBlob blob; 			
	Rect blob_rect;
	int current_pixel;
	int currentBlob = 0;
    vector<cvBlob> bloblist;

	fgmaskFiltered.convertTo(aux,CV_32SC1);																						// clear blob list
									
	for(int i=0;i<aux.rows;i++){
		for(int j=0;j<aux.cols;j++){
			current_pixel = aux.at<int>(i,j);																		// search for a foreground pixel
			if(255==current_pixel){
				area = floodFill(aux, Point(j,i), 1, &blob_rect, 0, 0, _connectivity);	
				blob = initBlob(currentBlob, blob_rect,area);													// initialize a new blob for connected foreground pixels
				bloblist.push_back(blob);																	// add detected blob to blob list
				currentBlob++;

			}
		}
	}

    return bloblist;
}



void KalmanTracker::_makeMeasurement(Mat frame){
		
	int morph_shape = 0; // Rectangular structuring element
	int morph_operation = 2; // Opening
	Mat element = getStructuringElement(morph_shape, Size(3,3));

	cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    _pMOG2->apply(frameGray, fgmask, _learningrate);
	morphologyEx(fgmask, fgmaskFiltered, morph_operation, element);
    vector<cvBlob> bloblist = _extractBlobs();

    int max_size = -1, maxh = -1, maxw = -1;
    measurement.at<float>(0) = 0;
	measurement.at<float>(1) = 0;

	if(_first_frame){
		_first_frame = false;
	}
	else{
		for(int i = 0; i < bloblist.size(); i++)	{
			cvBlob current_blob = bloblist[i];																    // get ith blob
			if((current_blob.w>_min_width)&&(current_blob.h>_min_height)){											// filter blobs by size
				int current_size = current_blob.area;
				if(current_size > max_size){                                                                        //if current blob is bigger than max_size, store current blob as the bigest
					measurement.at<float>(0) = current_blob.x + current_blob.w/2;
					measurement.at<float>(1) = current_blob.y + current_blob.h/2;
					max_size = current_size;
					maxh = current_blob.h;
					maxw = current_blob.w;
				}
			}
		}
	}

	measPt.x = measurement.at<float>(0);
	measPt.y = measurement.at<float>(1);
}

void KalmanTracker::_initKalman(){

	// measurement or observation matrix translates states into measures to allow for comparison between both domains... should not be changed
	// transition matrix defines velocity and acceleration models... should not modify
	// process noise defines the noise of the pbenomenon and can be modified
	// measurement noise defines how the measurement process may vary and can be modified
	// error covariance considers both measure and process

	
	if(_mode==0){			// constant velocity
		int stateSize = 4;
		int measSize = 2;
		_KF.init(stateSize,measSize,0);
		state = Mat::zeros(stateSize, 1, CV_32F);	
		prediction = Mat::zeros(stateSize, 1, CV_32F);	
		measurement = Mat::zeros(measSize, 1, CV_32F)-1;	
		_KF.measurementMatrix = (Mat_<float>(2,4) << 1,0,0,0, 0,0,1,0);
		_KF.transitionMatrix = (Mat_<float>(4,4) << 1,1,0,0, 0,1,0,0, 0,0,1,1, 0,0,0,1);
		
		_KF.processNoiseCov = (Mat_<float>(4,4) << 25,0,0,0, 0,900,0,0, 0,0,25,0, 0,0,0,10);
		_KF.measurementNoiseCov = (Mat_<float>(2,2) << 25,0, 0,1000);
		_KF.errorCovPost = (Mat_<float>(4,4) << 1000,0,0,0, 0,1000,0,0, 0,0,1000,0, 0,0,0,1000);
		

		

	}
	else{					// constant acceleration
		int stateSize = 6;
		int measSize = 2;
		_KF.init(stateSize,measSize,0);
		state = Mat::zeros(stateSize, 1, CV_32F);	
		prediction = Mat::zeros(stateSize, 1, CV_32F);	
		measurement = Mat::zeros(measSize, 1, CV_32F)-1;	
		_KF.measurementMatrix = (Mat_<float>(2,6) << 1,0,0,0,0,0, 0,0,0,1,0,0);
		_KF.transitionMatrix = (Mat_<float>(6,6) << 1,1,0.5,0,0,0, 0,1,1,0,0,0, 0,0,1,0,0,0, 0,0,0,1,1,0.5, 0,0,0,0,1,1, 0,0,0,0,0,1);
		
		_KF.processNoiseCov = (Mat_<float>(6,6) << 25,0,0,0,0,0, 0,10,0,0,0,0, 0,0,1,0,0,0, 0,0,0,25,0,0, 0,0,0,0,10,0, 0,0,0,0,0,0.1);
		_KF.measurementNoiseCov = (Mat_<float>(2,2) << 25,0, 0,25);	
		_KF.errorCovPost = (Mat_<float>(6,6) << 1000,0,0,0,0,0, 0,1000,0,0,0,0, 0,0,1000,0,0,0, 0,0,0,1000,0,0, 0,0,0,0,1000,0, 0,0,0,0,0,1000);

	}
}

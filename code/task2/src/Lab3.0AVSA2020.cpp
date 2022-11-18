//system libraries C/C++
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//Header ShowManyImages
#include "ShowManyImages.hpp"
//Header KalmanTracker
#include "KalmanTracker.hpp"

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

void printAllResults(Mat &img, vector<Point> points, Scalar color, int size, int border){
    
    for (int i = 0; i < points.size(); i++) { 
        circle(img, points[i],size,color,border);

        if(i > 0){
            line(img, points[i-1], points[i],color,2);
        }
    }
}

//main function
int main(int argc, char ** argv) {

    if (argc < 2){
		cout << "Missing argument." << endl;
        cout << "Example: ./Lab3.0AVSA2020 path/to/video1.mp4 path/to/video2.mp4" << endl;
        return -1;
	}
    
    Mat frame, results, results_all; // current Frame
    Point void_point(0, 0);
    string str;

    double t, acum_t; //variables for execution time
	int t_freq = getTickFrequency();

    string results_path = "./results";
    string makedir_cmd = "mkdir " + results_path;
    system(makedir_cmd.c_str());

    int num_videos = argc-1;
    cout << "Numvideos: " << num_videos << endl;

    for(int n_video=0; n_video < num_videos; n_video = n_video+1){
       
        str = to_string(n_video);
        makedir_cmd = "mkdir "+ results_path + "/sequence" + str;
		system(makedir_cmd.c_str());
        
        makedir_cmd = "mkdir "+ results_path + "/sequence" + str + "/frame";
		system(makedir_cmd.c_str()); 
        makedir_cmd = "mkdir "+ results_path + "/sequence" + str + "/fgnoise";
		system(makedir_cmd.c_str()); 
        makedir_cmd = "mkdir "+ results_path + "/sequence" + str + "/fgfilter";
		system(makedir_cmd.c_str()); 
        makedir_cmd = "mkdir "+ results_path + "/sequence" + str + "/kalman";
		system(makedir_cmd.c_str()); 

        string video_path = argv[n_video+1];
        
        VideoCapture cap;//reader to grab videoframes
        cout << "Accessing sequence at " << video_path << endl;

        cap.open(video_path);
        if (!cap.isOpened()) {
            cout << "Could not open video file " << video_path << endl;
            continue;
        }

        int it = 0; //Processed frames counter
		double acum_t = 0;  //Video processing time counter
        
        int mode = 1;
        KalmanTracker tracker(mode);

        vector<Point> estimations, predictions, measurements;

        for(;;){

            cap >> frame;

            frame.copyTo(results);
            frame.copyTo(results_all);
            //check if we achieved the end of the file (e.g. img.data is empty)
			if (!frame.data)
				break;

            //Time measurement
			t = (double)getTickCount();

            //Track biggest blob in frame
            tracker.track(frame);


            //cout << "Prediction: " << tracker.predPt << endl;
            //cout << "Measure: " << tracker.measPt << endl;
            //cout << "Estimation: " << tracker.statePt << endl;
			
            if (tracker.measPt != void_point)
                measurements.push_back(tracker.measPt);
            if (tracker.statePt != void_point)
			    estimations.push_back(tracker.statePt); 
            if (tracker.predPt != void_point)
			    predictions.push_back(tracker.predPt);            

            //Time measurement
            t = (double)getTickCount() - t;
			acum_t=+t;


            //SHOW RESULTS
            circle(results, tracker.predPt,10,CV_RGB(0,255,0),2);
            circle(results, tracker.measPt,10,CV_RGB(100,100,255),2);
            circle(results, tracker.statePt,10,CV_RGB(255,0,0),2);

            printAllResults(results_all, predictions,  CV_RGB(0,255,0),10,2);
            printAllResults(results_all, measurements, CV_RGB(100,100,255),8,2);
            printAllResults(results_all, estimations,  CV_RGB(255,0,0),4,2);
            
            putText(results, "Predictions", Point(10, 50), 1, 3, CV_RGB(0,255,0), 3);
            putText(results_all, "Predictions", Point(10, 50), 1, 3, CV_RGB(0,255,0), 3);
            putText(results, "Measurements", Point(10, 100), 1, 3, CV_RGB(100,100,255), 3);
            putText(results_all, "Measurements", Point(10, 100), 1, 3, CV_RGB(100,100,255), 3);
            putText(results, "Estimations", Point(10, 150), 1, 3, CV_RGB(255,0,0), 3);
            putText(results_all, "Estimations", Point(10, 150), 1, 3, CV_RGB(255,0,0), 3);

			string title =  "Lab3.0AVSA2020 | Frame - FgM - FgM filt | Blobs - Classes - Stat Classes | BlobsFil - ClassesFil - Stat ClassesFil";

			ShowManyImages(title, 6, frame, tracker.frameGray, tracker.fgmask, tracker.fgmaskFiltered, results, results_all);


            // STORE RESULTS
            string outFile1 = results_path + "/sequence" + str + "/frame/out"+ to_string(it) +".png";
            string outFile2 = results_path + "/sequence" + str + "/fgnoise/out"+ to_string(it) +".png";
            string outFile3 = results_path + "/sequence" + str + "/fgfilter/out"+ to_string(it) +".png";
            string outFile4 = results_path + "/sequence" + str + "/kalman/out"+ to_string(it) +".png";

            bool write_result1 = false;
	        write_result1 = imwrite(outFile1, frame);
	        if (!write_result1) printf("ERROR: Can't save fRAME mask.\n");
            bool write_result2 = false;
	        write_result2 = imwrite(outFile2, tracker.fgmask);
	        if (!write_result2) printf("ERROR: Can't save fg mask.\n");
            bool write_result3 = false;
	        write_result3 = imwrite(outFile3, tracker.fgmaskFiltered);
	        if (!write_result3) printf("ERROR: Can't save fg mask.\n");
            bool write_result4 = false;
	        write_result4 = imwrite(outFile4, results_all);
	        if (!write_result4) printf("ERROR: Can't save fg mask.\n");

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
		
			it++;
        } //video loop

        cout << it << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;
        cout << frame.size() <<endl;

        //release asll resources
        frame.release();
        destroyAllWindows();
       // (should stop till any key is pressed .. doesn't!!!!!)
        waitKey(0);
    }
    
    return 0;
}

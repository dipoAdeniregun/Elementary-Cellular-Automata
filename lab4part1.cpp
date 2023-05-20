#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
	assert(argc == 5);
		
	//-----------------------
	// Convert Command Line
	//-----------------------
			
	int nx = atoi(argv[1]);
    int initial_index = atoi(argv[2]);
    int rule = atoi(argv[3]);
    int maxiter = atoi(argv[4]);
    
    //---------------------------------
    // Generate the CA population
    //---------------------------------
    cv::Mat population(maxiter, nx, CV_8UC1);
    
    assert(initial_index >= 0 && initial_index < nx);
    
    //for easier printing we will make the zero state white
    for (unsigned int ix = 0; ix < nx; ix++)
    {
        population.at<uchar>(0,ix) = 255;
    }
    
    //and the one state black
    population.at<uchar>(0,initial_index) = 0;
    
    int neighbour_states[3];
    for (int iter = 0; iter < maxiter; iter++)
    {
        cv::namedWindow("Population", cv::WINDOW_AUTOSIZE );
        cv::imshow("Population", population);
        cv::waitKey(10);
        
        std::cout << "Iteration # " << iter << " of " << maxiter << std::endl;
        
        for (int ix = 0; ix < nx; ix++)
        {
            //This could be implemented far more efficiently on a sliding rule
            if (ix == 0)
            {
                neighbour_states[0] = (population.at<uchar>(iter, nx-1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
            }
            else if (ix == nx - 1)
            {
                neighbour_states[0] = (population.at<uchar>(iter,ix-1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter,0) == 0);
            }
            else
            {
                neighbour_states[0] = (population.at<uchar>(iter,ix-1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
            }
            
            //convert the neighbour states to an integer
            int neighbour_pattern_index = 0;
            neighbour_pattern_index |= (neighbour_states[0] << 2);
            neighbour_pattern_index |= (neighbour_states[1] << 1);
            neighbour_pattern_index |= (neighbour_states[2] << 0);
            
            //the next state is the "neighbour pattern index"th bit of the rule.
            int new_state = (rule & (1 << (neighbour_pattern_index))) != 0;
            
            //Uncomment if you want to see state conversion
            //std::cout << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
            //std::cout << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
            assert(new_state == 0  || new_state == 1);
            population.at<uchar>(iter+1,ix) = 255*(1-new_state);
        }
    }

     ostringstream converter;
     converter << "WrappingElementaryCA_" << nx << "_x_" << maxiter << "_Rule" << rule << ".jpg";
     imwrite(converter.str(),population);
    
    //cv::waitKey(10000);	//wait 10 seconds before closing image (or a keypress to close)
}

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "Utilities.h"
#include <mpi.h>

using namespace std;

int main(int argc, char** argv)
{
    int nproc = 1, rank = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	assert(argc == 5);
		
	//-----------------------
	// Convert Command Line
	//-----------------------
			
	int nx = atoi(argv[1]);
    int initial_index = atoi(argv[2]);
    int rule = atoi(argv[3]);
    int maxiter = atoi(argv[4]);
    ostringstream filename;
    filename << "Rank_" << rank << "_of_" << nproc << ".txt";

    ofstream myFile(filename.str()) ;
    
    if(rank == 0){
        int count;
        std::vector<int> local_start(nproc);
        std::vector<int> local_stop(nproc);
        std::vector<int> local_count(nproc);

        //---------------------------------
        // Generate the CA population
        //---------------------------------
        cv::Mat population(maxiter, nx+1, CV_8UC1);
        
        assert(initial_index >= 0 && initial_index < nx);
        
        //for easier printing we will make the zero state white
        for (unsigned int ix = 0; ix < nx; ix++)
        {
            population.at<uchar>(0,ix) = 255;
        }

        //and the one state black
        population.at<uchar>(0,initial_index) = 0;

        //set last item equals first item to help with sending halo rows
        population.at<uchar>(0, nx) = population.at<uchar>(0,0);

        // //print out original matrix
        // std::cout << "Original matrix: \n";
        // for(int irow = 0; irow< maxiter; irow++){
        //     for(int icol = 0; icol < nx+1; icol++){
        //         std::cout<< (int) population.at<uchar>(irow, icol) << ", ";
        //     }
        // }
        // std::cout << "\n";

        double start = MPI_Wtime();

        //generate local start stop and counts using parallel range taking a single halo row on either side into account 
        for(int irank = 0; irank < nproc; irank++){
            parallelRange(0, nx-1, irank, nproc, local_start[irank], local_stop[irank], local_count[irank]);

            if (irank == 0){
                local_count[irank] +=1;
                local_stop[irank] += 1;
            }
            else{
                local_start[irank] -= 1;
                local_count[irank] += 2;
                local_stop[irank] += 1;
            }
        }

        //scatter local counts for each rank in preparation for scatterv
        MPI_Scatter(&local_count[0], 1, MPI_INT, &count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //send data in first iteration of ECA
        MPI_Scatterv(&population.at<uchar>(0), &local_count[0], &local_start[0], MPI_BYTE, &population.at<uchar>(0), count, MPI_BYTE, 0, MPI_COMM_WORLD);

        // //print out original matrix
        myFile << "Matrix on rank " << rank << " after scatter: \n";
        // for(int irow = 0; irow< maxiter; irow++){
            myFile << (int) population.at<uchar>(0, nx-1) << ", ";
             for(int icol = 0; icol < count; icol++){
                 myFile << (int) population.at<uchar>(0, icol) << ", ";
             }
        // }
        myFile << std::endl;



        for (int iter = 0; iter < maxiter-1; iter++)
        {
            // cv::namedWindow("Population", cv::WINDOW_AUTOSIZE );
            // cv::imshow("Population", population);
            // cv::waitKey(10);
            
            //std::cout << "Iteration # " << iter << " of " << maxiter << " on rank "<< rank << std::endl;
            
            int neighbour_states[3];
            for (int ix = 0; ix < count - 1; ix++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if(ix == 0){
                    neighbour_states[0] = (population.at<uchar>(iter,nx-1) == 0);
                    neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
                    neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
                }
                else{
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
                myFile << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
                myFile << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
                assert(new_state == 0  || new_state == 1);
                population.at<uchar>(iter+1,ix) = 255*(1-new_state);

                
            }
                
                //sendrecv
                MPI_Status bottom_status, top_status;

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Sendrecv(&population.at<uchar>(iter+1,count-2), 1, MPI_BYTE, (rank+1)%nproc, 99, &population.at<uchar>(iter+1,nx-1), 1, MPI_BYTE, nproc-1, 99, MPI_COMM_WORLD, &top_status);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Sendrecv(&population.at<uchar>(iter+1, 0), 1, MPI_BYTE, nproc-1, 99, &population.at<uchar>(iter+1, count-1), 1, MPI_BYTE, (rank+1)%nproc, 99, MPI_COMM_WORLD, &bottom_status);
                MPI_Gatherv(&population.at<uchar>(iter+1,0), count-1, MPI_BYTE, &population.at<uchar>(iter+1,0), &local_count[0], &local_start[0], MPI_BYTE, 0, MPI_COMM_WORLD);

                
                myFile << "Matrix on rank " << rank << " after " << iter + 1<< " iterations: \n";
                // for(int irow = 0; irow< maxiter; irow++){
                myFile<< (int) population.at<uchar>(iter+1, nx-1) << ", ";
                for(int icol = 0; icol < count; icol++){
                        myFile<< (int) population.at<uchar>(iter+1, icol) << ", ";
                    }myFile << std::endl;
                // }
                myFile << "\n";
                myFile << "\n";
                myFile << "---------------------------------------------\n";
                myFile << "Rank " << rank << ": new halos sent\n";
                myFile << "---------------------------------------------\n";
                myFile << "left halo: " << (int)population.at<uchar>(iter+1, 0) << " right halo: " <<  (int)population.at<uchar>(iter+1, count-2) << "\n";
                myFile << "---------------------------------------------\n\n";

                myFile << "---------------------------------------------\n";
                myFile << "Rank " << rank << ": new halos received\n";
                myFile << "---------------------------------------------\n";
                myFile << "left halo: " <<(int) population.at<uchar>(iter+1, nx-1) << " right halo: " <<  (int)population.at<uchar>(iter+1, count-1) << "\n";
                myFile << "---------------------------------------------\n\n";
            
        }

        double end = MPI_Wtime();
        std::cout << "nproc = " << nproc << " problem size = " << nx <<  " Time elapsed = " << end - start << std::endl;

        //reset local start values to original value without halo items considered
        // for(int irank = 0; irank < nproc; irank++){
        //     parallelRange(0, nx-1, irank, nproc, local_start[irank], local_stop[irank], local_count[irank]);
        // }

        MPI_Barrier(MPI_COMM_WORLD);
        
        //std::cout << "DONE WITH GATHER" <<std::endl;

        cv::namedWindow("Population", cv::WINDOW_AUTOSIZE );
        cv::imshow("Population", population);
        //cv::waitKey(1000);

        ostringstream converter;
        converter << "PWElementaryCA_" <<   nx << "_x_" << maxiter << "_Rule" << rule <<"_rank" << rank<<".jpg";
        bool success = imwrite(converter.str(),population);
        if(success) std::cout << "image write successful on rank " << rank << std::endl;
        else std::cout << "image write unsuccessful on rank " << rank << std::endl;
        

    }

    else{
        int count;

        MPI_Scatter(NULL, 1, MPI_INT, &count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cv::Mat population(maxiter, count, CV_8UC1);

        //send data in first iteration of ECA
        MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE, &population.at<uchar>(0), count, MPI_BYTE, 0, MPI_COMM_WORLD);

        // //print out original matrix
        myFile << "Matrix on rank " << rank << " after scatter: \n";
        // for(int irow = 0; irow< maxiter; irow++){
        for(int icol = 0; icol < count; icol++){
                 myFile<< (int) population.at<uchar>(0, icol) << ", ";
             }
        // }
         myFile << "\n";

        for (int iter = 0; iter < maxiter-1; iter++)
        {
            
            
            //std::cout << "Iteration # " << iter << " of " << maxiter << " on rank "<< rank << std::endl;
            
            int neighbour_states[3];
            for (int ix = 1; ix < count - 1; ix++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                neighbour_states[0] = (population.at<uchar>(iter,ix-1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
                
                //convert the neighbour states to an integer
                int neighbour_pattern_index = 0;
                neighbour_pattern_index |= (neighbour_states[0] << 2);
                neighbour_pattern_index |= (neighbour_states[1] << 1);
                neighbour_pattern_index |= (neighbour_states[2] << 0);
                
                //the next state is the "neighbour pattern index"th bit of the rule.
                int new_state = (rule & (1 << (neighbour_pattern_index))) != 0;
                
                // Uncomment if you want to see state conversion
                myFile << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
                myFile << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
                assert(new_state == 0  || new_state == 1);
                population.at<uchar>(iter+1,ix) = 255*(1-new_state);

                
            }

                

                //sendrecv
                MPI_Status bottom_status, top_status;

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Sendrecv(&population.at<uchar>(iter+1, count-2), 1, MPI_BYTE, ((rank+1)%nproc), 99, &population.at<uchar>(iter+1, 0), 1, MPI_BYTE, rank-1, 99, MPI_COMM_WORLD, &top_status);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Sendrecv(&population.at<uchar>(iter+1, 1), 1, MPI_BYTE, rank-1, 99, &population.at<uchar>(iter+1, count-1), 1, MPI_BYTE, (rank+1)%nproc, 99, MPI_COMM_WORLD, &bottom_status);

                
                myFile << "Matrix on rank " << rank << " after " << iter + 1 << " iterations: \n";
                // for(int irow = 0; irow< maxiter; irow++){
                for(int icol = 0; icol < count; icol++){
                        myFile<< (int) population.at<uchar>(iter+1, icol) << ", ";
                    }myFile << std::endl;
                // }

                myFile << "\n";
                myFile << "---------------------------------------------\n";
                myFile << "Rank " << rank << ": new halos sent\n";
                myFile << "---------------------------------------------\n";
                myFile << "left halo: " << (int)population.at<uchar>(iter+1, 1) << " right halo: " <<  (int)population.at<uchar>(iter+1, count-2) << "\n";
                myFile << "---------------------------------------------\n\n";

                myFile << "\n";
                myFile << "---------------------------------------------\n";
                myFile << "Rank " << rank << ": new halos received\n";
                myFile << "---------------------------------------------\n";
                myFile << "left halo: " << (int)population.at<uchar>(iter+1, 0) << " right halo: " <<  (int)population.at<uchar>(iter+1, count-1) << "\n";
                myFile << "---------------------------------------------\n\n";

                MPI_Gatherv(&population.at<uchar>(iter+1, 1), count-2, MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
            
        }
        
        // ostringstream converter;
        // converter << "PWElementaryCA_" <<   nx << "_x_" << maxiter << "_Rule" << rule <<"_rank" << rank <<".jpg";
        // bool success = imwrite(converter.str(),population);
        // if(success) std::cout << "image write successful on rank " << rank << std::endl; 
        MPI_Barrier(MPI_COMM_WORLD);
        
        
    }

    
    
    // int neighbour_states[3];
    // for (int iter = 0; iter < maxiter; iter++)
    // {
    //     cv::namedWindow("Population", cv::WINDOW_AUTOSIZE );
    //     cv::imshow("Population", population);
    //     cv::waitKey(10);
        
    //     std::cout << "Iteration # " << iter << " of " << maxiter << std::endl;
        
    //     for (int ix = 0; ix < nx; ix++)
    //     {
    //         //This could be implemented far more efficiently on a sliding rule
    //         if (ix == 0)
    //         {
    //             neighbour_states[0] = (population.at<uchar>(iter, nx-1) == 0);
    //             neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
    //             neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
    //         }
    //         else if (ix == nx - 1)
    //         {
    //             neighbour_states[0] = (population.at<uchar>(iter,ix-1) == 0);
    //             neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
    //             neighbour_states[2] = (population.at<uchar>(iter,0) == 0);
    //         }
    //         else
    //         {
    //             neighbour_states[0] = (population.at<uchar>(iter,ix-1) == 0);
    //             neighbour_states[1] = (population.at<uchar>(iter,ix) == 0);
    //             neighbour_states[2] = (population.at<uchar>(iter,ix+1) == 0);
    //         }
            
    //         //convert the neighbour states to an integer
    //         int neighbour_pattern_index = 0;
    //         neighbour_pattern_index |= (neighbour_states[0] << 2);
    //         neighbour_pattern_index |= (neighbour_states[1] << 1);
    //         neighbour_pattern_index |= (neighbour_states[2] << 0);
            
    //         //the next state is the "neighbour pattern index"th bit of the rule.
    //         int new_state = (rule & (1 << (neighbour_pattern_index))) != 0;
            
    //         //Uncomment if you want to see state conversion
    //         //std::cout << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
    //         //std::cout << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
    //         assert(new_state == 0  || new_state == 1);
    //         population.at<uchar>(iter+1,ix) = 255*(1-new_state);
    //     }
    // }

     
    
    // //cv::waitKey(10000);	//wait 10 seconds before closing image (or a keypress to close)
    myFile.close();
    MPI_Finalize();
}

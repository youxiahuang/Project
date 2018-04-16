#include "camPose.h"
#include "RotationHelpers.h" 


namespace bn
{
    // The 3D mean shape vector of the PDM [x1,..,xn,y1,...yn,z1,...,zn]
	cv::Mat_<double> mean_shape;	
	// Principal components or variation bases of the model, 
	cv::Mat_<double> princ_comp;	
    // Eigenvalues (variances) corresponding to the bases
	cv::Mat_<double> eigen_values;	
        
    // Local parameters describing the non-rigid shape
	cv::Mat_<double>    params_local;
	// Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
	cv::Vec6d           params_global;
    
    bool tracking_initialised;
    
    cv::Size frame_size;
    float fx=-1, fy=-1, cx=-1, cy=-1;
    
    void GetParam(float &fx_, float &fy_, float &cx_, float &cy_)
    {
    	fx_ = fx;
    	fy_ = fx;
    	cx_ = cx;
    	cy_ = cy;
    }
    
    // Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row,col,type;
	
	//std::string tmp;
	//std::getline(stream, tmp);
	//row = atoi(tmp.c_str());
	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);
	
	switch(output_mat.type())
	{
		case CV_64FC1: 
		{
			cv::MatIterator_<double> begin_it = output_mat.begin<double>();
			cv::MatIterator_<double> end_it = output_mat.end<double>();
			
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
				//std::getline(stream, tmp);
				//*begin_it++ = atof(tmp.c_str());
			}
		}
		break;
		case CV_32FC1:
		{
			cv::MatIterator_<float> begin_it = output_mat.begin<float>();
			cv::MatIterator_<float> end_it = output_mat.end<float>();

			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32SC1:
		{
			cv::MatIterator_<int> begin_it = output_mat.begin<int>();
			cv::MatIterator_<int> end_it = output_mat.end<int>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_8UC1:
		{
			cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
			cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		default:
			printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__,__LINE__,output_mat.type()); abort();


	}
}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{	
	while(stream.peek() == '#' || stream.peek() == '\n'|| stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

void Read(string location)
{
  	
	ifstream pdmLoc(location, ios_base::in);

	SkipComments(pdmLoc);

	// Reading mean values
	ReadMat(pdmLoc,mean_shape);
	
	SkipComments(pdmLoc);

	// Reading principal components
	ReadMat(pdmLoc,princ_comp);
	
	SkipComments(pdmLoc);
	
	// Reading eigenvalues	
	ReadMat(pdmLoc,eigen_values);

}

    static void SetCameraIntrinsics(float fx_ =-1, float fy_=-1, float cx_=-1, float cy_=-1)
    {
        // If optical centers are not defined just use center of image
        if (cx == -1)
        {
            cx = frame_size.width / 2.0f;
            cy = frame_size.height/ 2.0f;
        }
        else
        {
            cx = cx_;
            cy = cy_;
        }
        // Use a rough guess-timate of focal length
        if (fx == -1)
        {
            fx = 500.0f * (frame_size.width / 640.0f);
            fy = 500.0f * (frame_size.height / 480.0f);

            fx = (fx + fy) / 2.0f;
            fy = fx;
        }
        else
        {
            fx = fx_;
            fy = fy_;
        }
    }
    
    int cam_pose_init(cv::Size frameSize_,string fname)
    {
        if(fname == ""){
            fname = "data/In-the-wild_aligned_PDM_68.txt";
        }
        Read(fname);
        
        frame_size = frameSize_;
        SetCameraIntrinsics();
        
        //mean_shape
        //princ_comp
        //cout << "mean_shape :" <<mean_shape<<endl; 
        //cout << "princ_comp :" <<princ_comp<<endl;
       
        params_local.create(princ_comp.cols, 1);
        params_local.setTo(0.0);
        // global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
        params_global = cv::Vec6d(0, 0, 0, 0, 0, 0);
        
        tracking_initialised = false;

		return 1;
    }
    
    //===========================================================================
// Compute the 3D representation of shape (in object space) using the local parameters
static void CalcShape3D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& p_local) 
{
	out_shape.create(mean_shape.rows, mean_shape.cols);
	out_shape = mean_shape + princ_comp*p_local;
}

    //===========================================================================
// provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
// This all assumes that the bounding box describes face from left outline to right outline of the face and chin to eyebrows
static void CalcParams(cv::Vec6d& out_params_global, const cv::Rect_<double>& bounding_box, const cv::Mat_<double>& params_local, const cv::Vec3d rotation = cv::Vec3d(0.0))
{

	// get the shape instance based on local params
	cv::Mat_<double> current_shape(mean_shape.size());

	CalcShape3D(current_shape, params_local);

	// rotate the shape
	cv::Matx33d rotation_matrix = Utilities::Euler2RotationMatrix(rotation);

	cv::Mat_<double> reshaped = current_shape.reshape(1, 3);

	cv::Mat rotated_shape = (cv::Mat(rotation_matrix) * reshaped);

	// Get the width of expected shape
	double min_x;
	double max_x;
	cv::minMaxLoc(rotated_shape.row(0), &min_x, &max_x);	

	double min_y;
	double max_y;
	cv::minMaxLoc(rotated_shape.row(1), &min_y, &max_y);

	double width = abs(min_x - max_x);
	double height = abs(min_y - max_y);

	double scaling = ((bounding_box.width / width) + (bounding_box.height / height)) / 2;

	// The estimate of face center also needs some correction
	double tx = bounding_box.x + bounding_box.width / 2;
	double ty = bounding_box.y + bounding_box.height / 2;

	// Correct it so that the bounding box is just around the minimum and maximum point in the initialised face	
	tx = tx - scaling * (min_x + max_x)/2;
    ty = ty - scaling * (min_y + max_y)/2;

	out_params_global = cv::Vec6d(scaling, rotation[0], rotation[1], rotation[2], tx, ty);
}

	int SetInitialized(const cv::Rect_<double>& bounding_box,bool initialized = true)
    {
        tracking_initialised = initialized;
        params_local.setTo(0);
        CalcParams(params_global, bounding_box, params_local);

		return 1;
    }
    
    
    static cv::Vec6d GetPose(cv::Mat_<double> &detected_landmarks)
    {
        if (!detected_landmarks.empty()&& params_global[0] != 0)
        {
            // This is used as an initial estimate for the iterative PnP algorithm
            double Z = fx / params_global[0];

            double X = ((params_global[4] - cx) * (1.0 / fx)) * Z;
            double Y = ((params_global[5] - cy) * (1.0 / fy)) * Z;

            // Correction for orientation

            // 2D points
            cv::Mat_<double> landmarks_2D = detected_landmarks; //特征点 68 个2维（x,y）

            //landmarks_2D = landmarks_2D.reshape(1, 2).t();

            // 3D points
            cv::Mat_<double> landmarks_3D;
            CalcShape3D(landmarks_3D, params_local); //特征点 68 个3维（x,y,z）

            landmarks_3D = landmarks_3D.reshape(1, 3).t();

            // Solving the PNP model

            // The camera matrix
            cv::Matx33d camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1); //相机内参

            cv::Vec3d vec_trans(X, Y, Z);  //平移
            cv::Vec3d vec_rot(params_global[1], params_global[2], params_global[3]); //旋转

            cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

            cv::Vec3d euler = Utilities::AxisAngle2Euler(vec_rot); //欧拉角

            return cv::Vec6d(vec_trans[0], vec_trans[1], vec_trans[2], euler[0], euler[1], euler[2]);
        }
        else
        {
            return cv::Vec6d(0, 0, 0, 0, 0, 0);
        }
    }

    cv::Vec6d get_pose(cv::Rect_<double>& Faces,cv::Mat_<double> &detected_landmarks,bool flag_,float fx_, float fy_, float cx_, float cy_)
    {
        if(fx_!=-1||fy_!=-1||cx_!=-1||cy_!=-1){
            SetCameraIntrinsics(fx_,fy_,cx_,cy_);
        }
        if(flag_ == true)
        {
            SetInitialized(Faces);
        }
        
        if(tracking_initialised != true){
            return cv::Vec6d(0, 0, 0, 0, 0, 0);
        }
        
        return GetPose(detected_landmarks);
    }
    
    int cam_pose_uninit(void)
    {
		return 1;
    }
}

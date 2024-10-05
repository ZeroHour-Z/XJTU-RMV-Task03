#include "windmill.hpp"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define PI 3.1415926
//vector<double> times;
//vector<double> angles;
struct WindmillCostFunctor
{
    WindmillCostFunctor(double time, double angle): 
        time_(time), angle_(angle) {}
    template <typename T>
    bool operator()(const T *const params, T *residual) const
    {
        T A = params[0];
        T w = params[1];
        T phi = params[2];
        T b = params[3];
        residual[0] = angle_ - cos(A / w * (cos(phi + PI / 2) - cos(w * time_ + phi + PI / 2)) + b * time_);//拟合值
        return true;
    }
private:
    const double time_;
    const double angle_;
};

int main()
{
    //google::InitGoogleLogging(argv[0]);
    double sum=0;
    for(int i=1;i<=10;i++)
    {
        //cout<<i<<endl;
        int64 start = getTickCount(),end;
        //int count = 0;
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        WINDMILL::WindMill wm((double)t.count());
        double t_start = (double)t.count();
        ceres::Problem problem;
        while (1)
        {
            //count++;
            t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            Mat src;
            src = wm.getMat((double)t.count());
            double t_now = (double)t.count();
            double times=(t_now-t_start)/1000;
            Mat gray;
            cvtColor(src, gray, COLOR_BGR2GRAY);
            Mat binary;
            threshold(gray, binary, 30, 255, THRESH_BINARY);
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); 
            int FlagR = -1, FlagHammer = -1;
            for (size_t i = 0; i < contours.size(); i++)
            {
                if (FlagR != -1 && FlagHammer != -1) break;
                if (hierarchy[i][3] == -1)//最外层轮廓
                {
                    if (contourArea(contours[i])< 500)//面积小于 500
                    {
                        FlagR = i;//是 FlagR
                        continue;
                    }
                    else if (contourArea(contours[i])< 10000)//500 ～ 10000
                        FlagHammer = hierarchy[i][2];//锤子
                }
            }
            Moments CenterR = moments(contours[FlagR]);
            Moments CenterHammer = moments(contours[FlagHammer]);
            Point2d R_pos(int(CenterR.m10 / CenterR.m00), int(CenterR.m01 / CenterR.m00));
            Point2d Hammer_pos(int(CenterHammer.m10 / CenterHammer.m00), int(CenterHammer.m01 / CenterHammer.m00));
            circle(src, R_pos, 2, Scalar(255, 255, 255), -1);
            circle(src, Hammer_pos, 2, Scalar(255, 255, 255), -1);
            //=======================================================//

            //imshow("Edges", edges);
            //imshow("windmill", src);

            //==========================代码区========================//
            double A = 1.785;
            double w = 0.884;
            double phi = 1.24;
            double b = 0.305;
            
            double params[4] = {A, w, phi, b};
            double angles = (Hammer_pos.x - R_pos.x)/norm(Hammer_pos - R_pos);
            ///sqrt(pow(Hammer_pos.x - R_pos.x, 2) + pow(Hammer_pos.y - R_pos.y, 2));
            //for (size_t i = 0; i < times.size(); ++i)
            //{
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WindmillCostFunctor, 1, 4>(new WindmillCostFunctor(times, angles)),nullptr,params);
            //problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WindmillCostFunctor, 1, 1,1,1,1>(new WindmillCostFunctor(times, angles)),nullptr,&b,&A,&w,&phi);
            //}
            //problem.AddParameterBlock(params, 4);
            // 配置求解器
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            //options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 25;
            problem.SetParameterLowerBound(params, 1, 0.5);
            problem.SetParameterUpperBound(params, 1, 1.89);
            problem.SetParameterLowerBound(params, 2, 0.24);
            problem.SetParameterUpperBound(params, 2, 1.24);
            problem.SetParameterLowerBound(params, 3, 0.5);
            problem.SetParameterUpperBound(params, 3, 1.38);
            // 运行求解器
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            // if (0.74575 < params[0] && params[0] < 0.82425 &&
            //     1.7898 < params[1] && params[1] < 1.97820 &&
            //     0.228 < params[2] && params[2] < 0.252 &&
            //     1.23975 < params[3] && params[3] < 1.37025)
            if (0.746 < params[0] && params[0] < 0.824 &&
                1.790 < params[1] && params[1] < 1.978 &&
                0.228 < params[2] && params[2] < 0.252 &&
                1.240 < params[3] && params[3] < 1.370)
            {
                end = getTickCount();
                sum+=(end - start) / getTickFrequency();
                break;
            }

            //waitKey(1);
            //=======================================================//    
            // cout << summary.FullReport() << "\n";
            // cout << "Estimated params:\n";
            // cout << "A: " << params[0] << "\n";
            // cout << "w: " << params[1] << "\n";
            // cout << "phi: " << params[2] << "\n";
            // cout << "b: " << params[3] << "\n";
        }   
    }
    cout<<sum/10<<endl;
}
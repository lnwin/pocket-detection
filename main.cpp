#include <QCoreApplication>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

struct Pt4 { float x,y,z,i; };
struct PlaneFit {
    Eigen::Vector3f n = {0,0,1}; // unit normal
    float d = 0.f; // n·x + d = 0
    std::vector<int> inliers;
    float rmse = 0.f;
};
// ---------------- IO ----------------
bool readTXT(const std::string& path, std::vector<Pt4>& pts){
    std::ifstream ifs(path); if(!ifs.is_open()) return false; pts.clear(); pts.reserve(1<<20);
    std::string line; double x,y,z,i;
    while(std::getline(ifs,line)){
        if(line.empty()) continue; std::istringstream ss(line);
        if(!(ss>>x>>y>>z>>i)) continue; pts.push_back({(float)x,(float)y,(float)z,(float)i});
    } return true;
}


bool writeTXT_labels(const std::string& path, const std::vector<Pt4>& pts, const std::vector<char>& lbl){
    std::ofstream ofs(path, std::ios::binary); if(!ofs.is_open()) return false;
    ofs.setf(std::ios::fixed); ofs.precision(4);
    for(size_t i=0;i<pts.size();++i){
        ofs<<pts[i].x<<" "<<pts[i].y<<" "<<pts[i].z<<" "<<(int)lbl[i]<<"\r\n";
    }
    ofs.flush(); return true;
}

static inline float absDist(const Eigen::Vector3f& p, const Eigen::Vector3f& n, float d){ return std::abs(n.dot(p)+d); }

PlaneFit refinePlane(const std::vector<Eigen::Vector3f>& P, const std::vector<int>& idx){
    PlaneFit pf; if(idx.empty()) return pf; Eigen::Vector3f c=Eigen::Vector3f::Zero(); for(int i:idx) c+=P[i]; c/=float(idx.size());
    Eigen::Matrix3f C=Eigen::Matrix3f::Zero(); for(int i:idx){ auto q=P[i]-c; C+=q*q.transpose(); }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C); Eigen::Vector3f n=es.eigenvectors().col(0).normalized(); if(n.z()<0) n=-n; float d=-n.dot(c);
    double se=0; for(int i:idx){ float e=n.dot(P[i])+d; se+=double(e)*e; } pf.n=n; pf.d=d; pf.inliers=idx; pf.rmse = idx.empty()?0.f:float(std::sqrt(se/double(idx.size()))); return pf;
}


PlaneFit ransacBestPlane(const std::vector<Eigen::Vector3f>& P, float thr, int max_iter)
{
    std::mt19937 rng(12345u); std::uniform_int_distribution<int> uni(0, int(P.size())-1);
    int best_cnt=-1; Eigen::Vector3f best_n(0,0,1); float best_d=0; std::vector<int> best_inliers;
    for(int it=0; it<max_iter; ++it){ int i=uni(rng), j=uni(rng), k=uni(rng); if(i==j||j==k||i==k){--it; continue;} auto &p1=P[i],&p2=P[j],&p3=P[k];
        Eigen::Vector3f n=(p2-p1).cross(p3-p1); float nn=n.norm(); if(nn<1e-6f){--it; continue;} n/=nn; if(n.z()<0) n=-n; float d=-n.dot(p1);
        std::vector<int> inliers; inliers.reserve(P.size()/4);
        for(int t=0;t<(int)P.size();++t) if(absDist(P[t],n,d)<=thr) inliers.push_back(t);
        if((int)inliers.size()>best_cnt){ best_cnt=(int)inliers.size(); best_n=n; best_d=d; best_inliers.swap(inliers);} }
    if(best_cnt<0){ return {}; }
    return refinePlane(P, best_inliers);
};
// ---------------- Align to Z ----------------
Eigen::Matrix3f rotFromAToB(const Eigen::Vector3f& a_raw, const Eigen::Vector3f& b_raw) {
    Eigen::Vector3f a=a_raw.normalized(), b=b_raw.normalized();
    float c = std::max(-1.0f, std::min(1.0f, a.dot(b)));
    if (c > 1.0f - 1e-6f) return Eigen::Matrix3f::Identity();
    if (c < -1.0f + 1e-6f) {
        Eigen::Vector3f axis = (std::abs(a.x()) < 0.9f) ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitY();
        axis = (axis - axis.dot(a)*a).normalized();
        Eigen::Matrix3f K; K<<0,-axis.z(),axis.y(), axis.z(),0,-axis.x(), -axis.y(),axis.x(),0;
        return Eigen::Matrix3f::Identity() + 2.0f*K*K; // 180°
    }
    Eigen::Vector3f v=a.cross(b); float s=v.norm(); Eigen::Matrix3f K; K<<0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0;
    return Eigen::Matrix3f::Identity() + K + K*K * ((1.0f - c)/(s*s));
}

// ---------------- 2D Geometry on XY ----------------
static inline float cross2d(const Eigen::Vector2f& o, const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return (a.x()-o.x())*(b.y()-o.y()) - (a.y()-o.y())*(b.x()-o.x());
}

std::vector<Eigen::Vector2f> convexHull(std::vector<Eigen::Vector2f> pts){
    if (pts.size() < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](auto& A, auto& B){ return (A.x()==B.x()) ? (A.y()<B.y()) : (A.x()<B.x()); });
    std::vector<Eigen::Vector2f> H; H.reserve(pts.size()*2);
    for (auto& p: pts){ while(H.size()>=2 && cross2d(H[H.size()-2], H.back(), p) <= 0) H.pop_back(); H.push_back(p);} // lower
    size_t t=H.size()+1; for(int i=(int)pts.size()-2;i>=0;--i){ auto& p=pts[i]; while(H.size()>=t && cross2d(H[H.size()-2], H.back(), p) <= 0) H.pop_back(); H.push_back(p);} // upper
    H.pop_back();
    // CCW
    double area=0; for(size_t i=0;i<H.size();++i){ auto& a=H[i]; auto& b=H[(i+1)%H.size()]; area += a.x()*b.y()-a.y()*b.x(); }
    if(area<0) std::reverse(H.begin(), H.end());
    return H;
}


std::vector<Eigen::Vector2f> shrinkHullRadial(const std::vector<Eigen::Vector2f>& hull, float margin){
    if(hull.size()<3 || margin<=0) return hull; Eigen::Vector2f c(0,0); for(auto& p:hull) c+=p; c/=float(hull.size());
    std::vector<Eigen::Vector2f> H; H.reserve(hull.size());
    for(auto& p:hull){ Eigen::Vector2f v=p-c; float r=v.norm(); if(r<=margin) H.push_back(c); else H.push_back(c + v*((r-margin)/r)); }
    return H;
}


bool pointInConvex(const std::vector<Eigen::Vector2f>& hull, const Eigen::Vector2f& p, float eps=1e-5f){
    if(hull.size()<3) return false; bool pos=false, neg=false;
    for(size_t i=0;i<hull.size();++i){ auto& a=hull[i]; auto& b=hull[(i+1)%hull.size()]; float c=cross2d(a,b,p); if(c>eps) pos=true; else if(c<-eps) neg=true; if(pos&&neg) return false; }
    return true;
}
void addHullPolylineAlignedToWorld(const std::vector<Eigen::Vector2f>& hull_xy,
                                   float step_mm,
                                   const Eigen::Matrix3f& R, // align rot
                                   const Eigen::Vector3f& t, // align trans (0,0,-z0)
                                   std::vector<Pt4>& pts_out,
                                   std::vector<char>& label_out,
                                   char hull_label=2)
{
    if (hull_xy.size()<2) return; Eigen::Matrix3f Rt = R.transpose();
    for(size_t i=0;i<hull_xy.size();++i){
        Eigen::Vector2f A=hull_xy[i]; Eigen::Vector2f B=hull_xy[(i+1)%hull_xy.size()];
        Eigen::Vector2f AB=B-A; float L=AB.norm(); if(L<=1e-6f) continue; int num=std::max(1,(int)std::ceil(L/std::max(1e-3f,step_mm)));
        for(int k=0;k<=num;++k){ float tt=float(k)/float(num); Eigen::Vector2f xy=A+tt*AB; Eigen::Vector3f q_align(xy.x(), xy.y(), 0.0f); Eigen::Vector3f p_world = Rt * (q_align - t);
            pts_out.push_back(Pt4{p_world.x(), p_world.y(), p_world.z(), 0.0f}); label_out.push_back(hull_label); }
    }
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    std::string in_path = "C:/Users/40582/Desktop/2025_08_13_16_14_15.txt";
    std::string out_path ="C:/Users/40582/Desktop/outPut.txt";
    // === 路径 & 参数（自行修改） ===
    float thr = 3.0f; // 判定在平面上的距离阈值（mm）——建议 0.6~1.2
    int iters = 2000; // RANSAC 迭代次数
    float margin = 2.0f*thr; // 凸包向内收缩量（mm）
    float tau_up = 0.7f*thr; // 上方容差（更严，剔除上方杂点）
    float tau_down = 5.0f*thr; // 下方最大允许（极深离群也丢弃）


    // === 读取 ===
    std::vector<Pt4> pts; if(!readTXT(in_path, pts)){ std::cerr<<"Fail to read "<<in_path<<"\n"; return 1; }
    if(pts.size()<3){ std::cerr<<"Too few points\n"; return 1; }


    // === RANSAC 拟合平面 ===
    std::vector<Eigen::Vector3f> P; P.reserve(pts.size()); for(auto&p:pts) P.emplace_back(p.x,p.y,p.z);
    PlaneFit pf = ransacBestPlane(P, thr, iters);
    float ratio = float(pf.inliers.size())/float(pts.size());
    std::cerr<<"[PLANE] inliers="<<pf.inliers.size()<<"/"<<pts.size()<<" ("<<ratio*100<<"%), rmse="<<pf.rmse<<"\n"
              <<" n="<<pf.n.transpose()<<", d="<<pf.d<<" (n·x + d = 0)\n";


    // === 对齐到 XY（使平面是 z=0） ===
    Eigen::Matrix3f R = rotFromAToB(pf.n, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f p0 = -pf.d * pf.n; // plane point
    Eigen::Vector3f p0r = R * p0; // rotated
    Eigen::Vector3f t(0,0,-p0r.z()); // translate z


    std::vector<Eigen::Vector3f> Palign; Palign.reserve(P.size());
    for(const auto& pw: P){ Eigen::Vector3f pr = R*pw; Palign.emplace_back(pr.x(), pr.y(), pr.z()+t.z()); }


    // === 基于对齐坐标的“核心平面候选” → 最大连通区域 → 凸包 ===
    float thr_core = 0.6f * thr;
    float pixel_mm_core = std::max(0.25f, 0.5f*thr); // 避免与后面 pixel_mm 重名

    float xmin=1e30f,xmax=-1e30f,ymin=1e30f,ymax=-1e30f;
    std::vector<size_t> core_ids; core_ids.reserve(Palign.size());
    for(size_t i=0;i<Palign.size();++i){
        float z = Palign[i].z();
        if(std::abs(z) <= thr_core){
            core_ids.push_back(i);
            xmin = std::min(xmin, Palign[i].x());
            xmax = std::max(xmax, Palign[i].x());
            ymin = std::min(ymin, Palign[i].y());
            ymax = std::max(ymax, Palign[i].y());
        }
    }

    std::vector<Eigen::Vector2f> hull; // **作用域在外，后续要用**

    if(core_ids.size() < 100){
        // 兜底：直接用 RANSAC inliers
        std::vector<Eigen::Vector2f> inlierXY; inlierXY.reserve(pf.inliers.size());
        for(int idx: pf.inliers){ const auto& q = Palign[idx]; inlierXY.emplace_back(q.x(), q.y()); }
        hull = convexHull(inlierXY);
        hull = shrinkHullRadial(hull, 2.0f*thr);
    } else {
        // 建栅格 → 连通域 → 取最大连通片 → 轮廓 → 凸包
        float pad = 3.0f * thr; xmin-=pad; xmax+=pad; ymin-=pad; ymax+=pad;
        int W = std::max(1, (int)std::ceil((xmax-xmin)/pixel_mm_core));
        int H = std::max(1, (int)std::ceil((ymax-ymin)/pixel_mm_core));
        cv::Mat mask(H, W, CV_8U, cv::Scalar(0));

        for(size_t id : core_ids){
            int x = (int)std::round((Palign[id].x() - xmin)/pixel_mm_core);
            int y = (int)std::round((Palign[id].y() - ymin)/pixel_mm_core);
            if(x>=0 && y>=0 && x<W && y<H) mask.at<uint8_t>(y,x) = 255;
        }

        int k = std::max(1, (int)std::round(1.0f * thr / pixel_mm_core));
        cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*k+1, 2*k+1));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, ker);

        cv::Mat labels, stats, centroids;
        int ncc = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
        int best = -1; int bestArea = -1;
        for(int lbl=1; lbl<ncc; ++lbl){
            int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
            if(area > bestArea){ bestArea = area; best = lbl; }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::Mat binMain = (labels == best);
        cv::findContours(binMain, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Point> cc;
        if(!contours.empty()){
            int bi=0; double bA=0;
            for(int i=0;i<(int)contours.size();++i){ double A=fabs(cv::contourArea(contours[i])); if(A>bA){bA=A; bi=i;} }
            cc = contours[bi];
        }

        std::vector<Eigen::Vector2f> inlierXY; inlierXY.reserve(cc.size());
        if(!cc.empty()){
            std::vector<cv::Point> ccHull; cv::convexHull(cc, ccHull);
            for(auto& p : ccHull){
                float X = xmin + p.x * pixel_mm_core;
                float Y = ymin + p.y * pixel_mm_core;
                inlierXY.emplace_back(X, Y);
            }
        } else {
            for(size_t id: core_ids) inlierXY.emplace_back(Palign[id].x(), Palign[id].y());
        }

        hull = convexHull(inlierXY);
        hull = shrinkHullRadial(hull, 2.0f*thr);
    }

// ===（可选）掩膜腐蚀版边界（更狠的裁边）===
#if USE_OPENCV_MASK
    float erosion_mm = std::max(2.0f*thr, 1.0f); // 腐蚀厚度（mm）
    float pixel_mm = std::max(0.25f, 0.5f*thr); // 栅格分辨率（mm/px）
    auto mask = makeXYMaskEroded(hull, erosion_mm, pixel_mm);
#endif


    // === 分类与裁剪 ===
    std::vector<Pt4> pts_keep; pts_keep.reserve(pts.size());
    std::vector<char> label_keep; label_keep.reserve(pts.size());


    for(size_t i=0;i<pts.size();++i){
        const auto& q = Palign[i];
        Eigen::Vector2f xy(q.x(), q.y());
#if USE_OPENCV_MASK
        if(!xy_in_mask(mask, xy.x(), xy.y())) continue; // 掩膜外 → 丢弃
#else
        if(!pointInConvex(hull, xy)) continue; // 凸包外 → 丢弃
#endif
        float z = q.z(); float ad = std::abs(z);
        if (z > tau_up) continue; // 上方杂点 → 丢弃
        if (z < -tau_down) continue; // 极深离群 → 丢弃
        if (ad <= thr) { pts_keep.push_back(pts[i]); label_keep.push_back(1); } // 平面(绿)
        else { pts_keep.push_back(pts[i]); label_keep.push_back(0); } // 凹槽(下方)
    }


    // === 画凸包折线（标签=2），写回到世界坐标 ===
    float line_step_mm = std::max(0.5f, std::min(2.0f, thr));
    addHullPolylineAlignedToWorld(hull, line_step_mm, R, t, pts_keep, label_keep, 2);


    // === 导出 ===
    if(!writeTXT_labels(out_path, pts_keep, label_keep)){ std::cerr<<"Fail to write "<<out_path<<"\n"; return 1; }
    std::cerr<<"Done: "<<out_path<<" (kept "<<pts_keep.size()<<" / "<<pts.size()<<" points)\n";
    return a.exec();

}




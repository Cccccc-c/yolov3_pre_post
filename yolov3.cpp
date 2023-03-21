#include <opencv2/opencv.hpp>
#include <iostream>
#include <cfenv>

const int resize_h = 640;
const int resize_w = 640;
const float nms_thresh = 0.45;
const float conf_thresh = 0.5;
  static constexpr int ANCHOR_NUM = 3;
const std::vector<float> biases = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0,
               119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
const std::vector<float> stride = {8.0, 16.0, 32.0};
const std::vector<int> output_sizeh = {80, 40, 20};
const std::vector<int> output_sizew = {80, 40, 20};

const int class_num = 80;
const std::vector<std::string> class_name = {"person",        "bicycle",      "car",
                   "motorcycle",    "airplane",     "bus",
                   "train",         "truck",        "boat",
                   "traffic light", "fire hydrant", "stop sign",
                   "parking meter", "bench",        "bird",
                   "cat",           "dog",          "horse",
                   "sheep",         "cow",          "elephant",
                   "bear",          "zebra",        "giraffe",
                   "backpack",      "umbrella",     "handbag",
                   "tie",           "suitcase",     "frisbee",
                   "skis",          "snowboard",    "sports ball",
                   "kite",          "baseball bat", "baseball glove",
                   "skateboard",    "surfboard",    "tennis racket",
                   "bottle",        "wine glass",   "cup",
                   "fork",          "knife",        "spoon",
                   "bowl",          "banana",       "apple",
                   "sandwich",      "orange",       "broccoli",
                   "carrot",        "hot dog",      "pizza",
                   "donut",         "cake",         "chair",
                   "couch",         "potted plant", "bed",
                   "dining table",  "toilet",       "tv",
                   "laptop",        "mouse",        "remote",
                   "keyboard",      "cell phone",   "microwave",
                   "oven",          "toaster",      "sink",
                   "refrigerator",  "book",         "clock",
                   "vase",          "scissors",     "teddy bear",
                   "hair drier",    "toothbrush"};

struct DetectYolov3 {
  float x;
  float y;
  float w;
  float h;
  float conf;
};
struct Detect2dObject {
  float x;
  float y;
  float w;
  float h;
  float class_prob;
  int class_idx;
};

template <typename T>
static inline bool compObject(const T &lhs, const T &rhs) {
  return lhs.class_prob > rhs.class_prob;
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

std::vector<int8_t> loadBin(const std::string &fname) {
  // get file length, allocate data buf
  std::fstream f(fname, std::ios_base::in | std::ios_base::binary);
  f.seekg(0, std::ios::end);
  auto fsize = f.tellg();
  f.seekg(0, std::ios::beg);

  // read file contents
  std::vector<int8_t> buf(fsize);
  f.read((char *)buf.data(), buf.size());
  f.close();

  return buf;
}

void saveBin(const std::string &fname, const char *data, uint64_t size) {
  // write into file
  std::ofstream f;
  f.open(fname, std::ios::binary | std::ios::out);

  // chkOpen(f, fname);

  f.write(data, size);
  f.close();
}

void preProcess(cv::Mat &resize_img,const float input_scale,std::vector<uint8_t>& npu_input) {
  // cv::Mat resize_img;
  // cv::resize(src_image,resize_img,cv::Size(resize_h,resize_w));

  cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

  uint8_t *src_data = resize_img.data;
  // uint8_t *input = npu_input.data();
  for(int i=0;i < 3 * resize_w * resize_h;i++)
  {
      // int data = floor((src_data[i] / 255.) * input_scale + 0.5);
      int data = static_cast<int>(nearbyint(src_data[i] / 255. / input_scale));
      if(data>255) data=255;
      if(data<0) data=0;
      npu_input[i] = (uint8_t)data;
  }
}

float iou(const cv::Rect &a, const cv::Rect &b) {
  float inter_area = (a & b).area();
  float union_area = a.area() + b.area() - inter_area;

  return inter_area / union_area;
}

void nms2d(std::map<int, std::list<Detect2dObject>> &m, float nms_thresh) {
  for (auto &e : m) {
    auto &l = e.second;
    l.sort(compObject<Detect2dObject>);

    // compute iou
    for (auto it = l.begin(); it != l.end(); it++) {
      cv::Rect a(it->x, it->y, it->w, it->h);
      auto cursor = it;
      std::advance(cursor, 1);

      for (; cursor != l.end();) {
        cv::Rect b(cursor->x, cursor->y, cursor->w, cursor->h);
        if (iou(a, b) > nms_thresh) {
          cursor = l.erase(cursor);
        } else {
          cursor++;
        }
      }
    }
  }
}

std::map<int,std::list<Detect2dObject>> filterBox(const float* output_scale,std::vector<std::vector<int8_t>>& npu_outputs) {
  std::map<int,std::list<Detect2dObject>> m;
  for (size_t n = 0; n < npu_outputs.size(); n++) {
    std::vector<float> output(npu_outputs[n].size());
    // printf("%f \n",output_scale[n]);
    for(int i=0;i<npu_outputs[n].size();i++){
        output[i] = npu_outputs[n][i]*output_scale[n];
    }
    // saveBin(std::to_string(n)+".bin",(const char *)output.data(),output.size()*sizeof(float));

    char *data =  (char *)output.data();
    int ptr_stride = sizeof(DetectYolov3) + sizeof(float) * class_num;

    int output_id = n;

    for (auto h = 0; h < output_sizeh[n]; h++) {
      for (auto w = 0; w < output_sizew[n]; w++) {
        for (auto anchor = 0; anchor < ANCHOR_NUM; anchor++) {
          
          auto idx = h * output_sizew[n] * ANCHOR_NUM + w * ANCHOR_NUM + anchor;
          char *obj = data + ptr_stride * idx;
          auto *yolov3_obj = (DetectYolov3 *)obj;
          std::vector<float> score(class_num);
          memcpy(score.data(), obj + sizeof(DetectYolov3),
                 sizeof(float) * class_num);

          auto it = std::max_element(score.begin(), score.end());
          auto class_id = std::distance(score.begin(), it);
          // NOVA_CHECK(class_id < class_num);
          auto sig_conf = sigmoid(yolov3_obj->conf) * sigmoid(score[class_id]);
          if (sig_conf < conf_thresh) {
            continue;
          }
          // construct an object according to current obj[i]
          Detect2dObject tmp;
          tmp.x = (sigmoid(yolov3_obj->x) * 2 - 0.5 + w) * stride[n];
          tmp.y = (sigmoid(yolov3_obj->y) * 2 - 0.5 + h) * stride[n];
          tmp.w = pow(sigmoid(yolov3_obj->w) * 2, 2) *
                  biases[output_id * 2 * ANCHOR_NUM + 2 * anchor + 0];
          tmp.h = pow(sigmoid(yolov3_obj->h) * 2, 2) *
                  biases[output_id * 2 * ANCHOR_NUM + 2 * anchor + 1];
          tmp.class_prob = sig_conf;
          tmp.class_idx = class_id;

          // move tmp Object into m std::map
          if (m.find(class_id) == m.end()) {
            m[class_id] =std::list<Detect2dObject>{std::move(tmp)};
          } else {
            m[class_id].emplace_back(std::move(tmp));
          }
        }
      }
    }
  }

  return m;
}

std::map<int,std::list<Detect2dObject>> postProcess(const float* output_scale,std::vector<std::vector<int8_t>>& npu_outputs) {
  auto m = filterBox(output_scale,npu_outputs);
  
  nms2d(m, nms_thresh);
  
  return m;
}

float getMinResizeScale(const cv::Mat &src, int dst_h, int dst_w) {
  auto src_h = src.rows;
  auto src_w = src.cols;

  auto scale_h = dst_h * 1.0 / src_h;
  auto scale_w = dst_w * 1.0 / src_w;

  return std::min(scale_h, scale_w);
}

cv::Point2i getOriginPoint(const cv::Mat &src, const cv::Mat &dst,
                           const cv::Point2i &dst_point) {
  auto min_scale = getMinResizeScale(src, dst.rows, dst.cols);
  auto real_dst_h = static_cast<int>(src.rows * min_scale);
  auto real_dst_w = static_cast<int>(src.cols * min_scale);
  auto x_offset = (dst.cols - real_dst_w) / 2;
  auto y_offset = (dst.rows - real_dst_h) / 2;

  cv::Point2i src_point;

  src_point.x = static_cast<int>(src.cols * dst_point.x * 1.f / dst.cols);
  src_point.y = static_cast<int>(src.rows * dst_point.y * 1.f / dst.rows);

  src_point.x = std::min(std::max(0, src_point.x), src.cols - 1);
  src_point.y = std::min(std::max(0, src_point.y), src.rows - 1);

  return src_point;
}

void draw2dBox(const std::map<int, std::list<Detect2dObject>> &m,
               cv::Mat &origin_frame, cv::Mat &resize_frame,
              const std::vector<std::string> &class_name) {
  for (auto &e : m) {
    auto &result = e.second;

    for (auto it = result.begin(); it != result.end(); it++) {
      cv::Point2i dst_tl_point(static_cast<int>(it->x - it->w * 0.5),
                               static_cast<int>(it->y - it->h * 0.5));
      auto src_tl_point =
          getOriginPoint(origin_frame, resize_frame, dst_tl_point);

      cv::Point2i dst_br_point(static_cast<int>(it->x + it->w * 0.5),
                               static_cast<int>(it->y + it->h * 0.5));
      auto src_br_point =
          getOriginPoint(origin_frame, resize_frame, dst_br_point);

      // draw box
      cv::rectangle(origin_frame, cv::Rect(src_tl_point, src_br_point),
                    cv::Scalar(0, 255, 255));

      cv::putText(origin_frame, class_name[it->class_idx], src_tl_point,
                  cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
    }
  }
}

int main(){
  fesetround(FE_TONEAREST);

  cv::Mat src = cv::imread("./data/car.jpg");
  cv::Mat resize_img;
  cv::resize(src,resize_img,cv::Size(resize_h,resize_w));

  std::vector<uint8_t> npu_input(3 * resize_w * resize_h);
  float input_scale = 0.003921568859368563;
  std::vector<float> output_scale = {0.12761929631233215,0.11434315145015717,0.13450734317302704};

  preProcess(resize_img,input_scale,npu_input);
  saveBin("./data/npu_input.bin",(const char *)npu_input.data(),3 * resize_w * resize_h);

  // TODO:: npu_run

  std::vector<std::vector<int8_t>> npu_outputs;
  npu_outputs.push_back(loadBin("./data/80x80x255.bin"));
  npu_outputs.push_back(loadBin("./data/40x40x255.bin"));
  npu_outputs.push_back(loadBin("./data/20x20x255.bin"));

  std::map<int,std::list<Detect2dObject>> m = postProcess(output_scale.data(),npu_outputs);
  draw2dBox(m, src, resize_img, class_name);
  cv::imwrite("./data/result.jpg",src);
  return 0;
}


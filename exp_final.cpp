#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <map>

// ==================== State Machine ====================
enum class State {
    INIT = 0,
    APPROACH_OBS_1 = 1,
    BLIND_SPOT_HOLD_1 = 2,
    TURN_RIGHT_TO_GAP = 3,
    APPROACH_GAP = 4,
    DIGIT_SEARCH = 5,
    DIGIT_TRACK = 6,
    FINISHED = 99
};

std::string stateToString(State s) {
    switch(s) {
        case State::INIT: return "INIT";
        case State::APPROACH_OBS_1: return "APPROACH_OBS_1";
        case State::BLIND_SPOT_HOLD_1: return "BLIND_SPOT_HOLD_1";
        case State::TURN_RIGHT_TO_GAP: return "TURN_RIGHT_TO_GAP";
        case State::APPROACH_GAP: return "APPROACH_GAP";
        case State::DIGIT_SEARCH: return "DIGIT_SEARCH";
        case State::DIGIT_TRACK: return "DIGIT_TRACK";
        case State::FINISHED: return "FINISHED";
        default: return "UNKNOWN";
    }
}

// ==================== Lane Detection Helper ====================
int calculateLaneCenter(const cv::Mat& binaryImage) {
    int height = binaryImage.rows;
    int width = binaryImage.cols;
    int roi_start_row = static_cast<int>(height * 0.6);
    int roi_end_row = static_cast<int>(height * 0.9);
    int roi_x_min = static_cast<int>(width * 0.4);
    int roi_x_max = static_cast<int>(width * 0.6);
    int num_scanlines = 5;
    int best_target_x = -1;
    
    for (int i = 0; i < num_scanlines; ++i) {
        int y = roi_start_row + (roi_end_row - roi_start_row) * i / (num_scanlines - 1);
        const uchar* row_ptr = binaryImage.ptr<uchar>(y);
        int current_gap_start = -1;
        int best_row_gap_center = -1;
        double best_row_gap_score = -1e9;
        
        for (int x = roi_x_min; x < roi_x_max; ++x) {
            int pixel = row_ptr[x];
            if (pixel == 0) { 
                if (current_gap_start == -1) current_gap_start = x;
            } else {
                if (current_gap_start != -1) {
                    int gap_width = x - current_gap_start;
                    if (gap_width > 20) {
                        int gap_center = current_gap_start + gap_width / 2;
                        double dist = std::abs(gap_center - (width / 2));
                        double score = gap_width - (0.5 * dist);
                        if (score > best_row_gap_score) {
                            best_row_gap_score = score;
                            best_row_gap_center = gap_center;
                        }
                    }
                    current_gap_start = -1;
                }
            }
        }
        if (current_gap_start != -1) {
            int gap_width = roi_x_max - current_gap_start;
            if (gap_width > 20) {
                int gap_center = current_gap_start + gap_width / 2;
                double dist = std::abs(gap_center - (width / 2));
                double score = gap_width - (0.5 * dist);
                if (score > best_row_gap_score) {
                    best_row_gap_score = score;
                    best_row_gap_center = gap_center;
                }
            }
        }
        if (best_row_gap_center != -1) {
            best_target_x = best_row_gap_center;
            break; 
        }
    }
    return best_target_x; 
}

// ==================== Robot Controller ====================
class RobotController {
public:
    RobotController() : 
        state_(State::INIT),
        blind_spot_counter_(0),
        turn_counter_(0),
        alignment_counter_(0),
        align_before_turn_(false),
        final_forward_active_(false),
        final_turn_active_(false),
        final_forward_counter_(0),
        final_turn_counter_(0),
        last_lane_target_x_(-1),
        SAFE_PASS_LIMIT_(100),
        MIN_TURN_DURATION_(15),
        FRAME_WIDTH_(1280),
        KP_(0.002),
        speed_init_(0.1),
        speed_normal_(0.2),
        speed_turn_(0.1),
        angular_turn_right_(-0.3),
        angular_turn_left_(0.3),
        speed_stop_(0.0),
        turn1_frames_(40),
        move_frames_(125),
        turn2_frames_(26),
        enter_left_min_(0.5),
        enter_right_min_(0.3),
        exit_left_min_(0.6),
        exit_right_max_(0.1),
        final_forward_frames_(10),
        final_turn_frames_(30),
        final_forward_linear_(0.0),
        final_forward_angular_(0.0),
        final_turn_angular_(-0.3),
        final_exit_ratio_max_(0.02),
        phase4_move_frames_(0),
        phase5_turn_frames_(0),
        digit_search_angular_(0),
        digit_track_kp_turn_(-0.0045),
        digit_track_kp_dist_(0.000015),
        selecting_roi_(false),
        best_digit_match_val_(-1.0),
        best_digit_label_(""),
        is_goal_reached_(false)
    {}
    
    void loadTemplates(const std::string& path) {
        template_path_ = path;
        std::vector<std::string> labels = {"0", "1", "2"};
        for (const auto& label : labels) {
            std::string filename = path + "template_" + label + ".jpg";
            cv::Mat templ = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (!templ.empty()) {
                templates_[label] = templ;
                cv::Mat temp_thresh;
                cv::threshold(templ, temp_thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
                template_areas_[label] = cv::countNonZero(temp_thresh);
                ROS_INFO("Loaded template: %s (Area: %d)", filename.c_str(), template_areas_[label]);
            } else {
                ROS_WARN("Template %s not found!", filename.c_str());
            }
        }
    }

    void onMouse(int event, int x, int y, int flags, const cv::Mat& current_gray_frame) {
        if (state_ != State::DIGIT_SEARCH && state_ != State::DIGIT_TRACK) return; 

        if (selecting_roi_) {
            roi_selector_.x = std::min(x, origin_.x);
            roi_selector_.y = std::min(y, origin_.y);
            roi_selector_.width = std::abs(x - origin_.x);
            roi_selector_.height = std::abs(y - origin_.y);
        }
        switch (event) {
            case cv::EVENT_LBUTTONDOWN:
                origin_ = cv::Point(x, y);
                roi_selector_ = cv::Rect(x, y, 0, 0);
                selecting_roi_ = true;
                break;
            case cv::EVENT_LBUTTONUP:
                selecting_roi_ = false;
                break;
        }
    }

    void handleDebugKeys(int key, const cv::Mat& gray) {
        if (key == '0' || key == '1' || key == '2') {
            if (roi_selector_.width > 10 && roi_selector_.height > 10) {
                cv::Mat templ = gray(roi_selector_).clone();
                std::string label(1, (char)key);
                templates_[label] = templ;
                cv::Mat temp_thresh;
                cv::threshold(templ, temp_thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
                template_areas_[label] = cv::countNonZero(temp_thresh);
                std::string filename = template_path_ + "template_" + label + ".jpg";
                cv::imwrite(filename, templ);
                ROS_INFO("Saved template %c (Area: %d)", (char)key, template_areas_[label]);
                roi_selector_ = cv::Rect();
            }
        } else if (key == 'r') {
            state_ = State::INIT; 
            ROS_INFO("Reset to INIT");
        }
    }

    void process(const cv::Mat& gray_frame,
                 const std::vector<std::pair<int, double>>& objects,
                 double left_box_ratio,
                 double right_box_ratio,
                 double final_left_ratio,
                 double final_right_ratio,
                 int& lane_target_x,
                 double& linear_x,
                 double& angular_z) {
        
        linear_x = 0.0;
        angular_z = 0.0;
        int center_x = FRAME_WIDTH_ / 2;
        
        if (state_ != State::DIGIT_SEARCH && state_ != State::DIGIT_TRACK && state_ != State::FINISHED) {
            if (lane_target_x >= 0) {
                if (last_lane_target_x_ < 0) last_lane_target_x_ = lane_target_x;
                double alpha = 0.4;
                lane_target_x = static_cast<int>(alpha * lane_target_x + (1.0 - alpha) * last_lane_target_x_);
                last_lane_target_x_ = lane_target_x;
            } else {
                if (last_lane_target_x_ >= 0) lane_target_x = last_lane_target_x_;
                else lane_target_x = center_x;
            }
        }

        std::pair<int, double> max_obj(0, 0);
        if (!objects.empty()) {
            max_obj = *std::max_element(objects.begin(), objects.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
        }
        int cx = max_obj.first;
        double area = max_obj.second;

        switch (state_) {
            case State::INIT:
            {
                int error = center_x - lane_target_x;
                angular_z = KP_ * error;
                angular_z = std::max(std::min(angular_z, 0.4), -0.4);
                if (std::abs(error) < 20) linear_x = speed_normal_;
                else linear_x = speed_init_;

                if (area > 40000 && cx > center_x) {
                    state_ = State::APPROACH_OBS_1;
                    ROS_INFO("State -> APPROACH_OBS_1");
                }
                break;
            }

            case State::APPROACH_OBS_1:
            {
                int error = center_x - lane_target_x;
                angular_z = KP_ * error;
                angular_z = std::max(std::min(angular_z, 0.4), -0.4);
                linear_x = speed_normal_;
                if (left_box_ratio > enter_left_min_ && right_box_ratio > enter_right_min_) {
                    state_ = State::BLIND_SPOT_HOLD_1;
                    ROS_INFO("State -> BLIND_SPOT_HOLD_1");
                }
                break;
            }

            case State::BLIND_SPOT_HOLD_1:
            {
                int error = center_x - lane_target_x;
                angular_z = KP_ * error;
                angular_z = std::max(std::min(angular_z, 0.4), -0.4);
                linear_x = speed_normal_;
                if (left_box_ratio > exit_left_min_ && right_box_ratio < exit_right_max_) {
                    state_ = State::TURN_RIGHT_TO_GAP;
                    turn_counter_ = 0;
                    ROS_INFO("State -> TURN_RIGHT_TO_GAP");
                }
                blind_spot_counter_++;
                if (blind_spot_counter_ > 300) {
                    state_ = State::TURN_RIGHT_TO_GAP;
                    turn_counter_ = 0;
                    ROS_INFO("State -> TURN_RIGHT_TO_GAP (Timeout)");
                }
                break;
            }

            case State::TURN_RIGHT_TO_GAP:
            {
                turn_counter_++;
                int phase1_end = turn1_frames_;
                int phase2_end = phase1_end + move_frames_;
                int phase3_end = phase2_end + turn2_frames_;
                int phase4_end = phase3_end + phase4_move_frames_;
                int phase5_end = phase4_end + phase5_turn_frames_;

                if (turn_counter_ <= phase1_end) {
                    linear_x = 0.0; angular_z = angular_turn_right_;
                } else if (turn_counter_ <= phase2_end) {
                    linear_x = speed_normal_; angular_z = 0.0;
                } else if (turn_counter_ <= phase3_end) {
                    linear_x = 0.0; angular_z = angular_turn_left_;
                } else if (turn_counter_ <= phase4_end) {
                    linear_x = speed_normal_; angular_z = 0.0;
                } else if (turn_counter_ <= phase5_end) {
                    linear_x = 0.0; angular_z = angular_turn_right_;
                } else {
                    state_ = State::APPROACH_GAP;
                    turn_counter_ = 0;
                    ROS_INFO("TURN_RIGHT_TO_GAP -> APPROACH_GAP");
                }
                break;
            }

            case State::APPROACH_GAP:
            {
                if (final_turn_active_) {
                    linear_x = 0.0;
                    angular_z = final_turn_angular_;
                    final_turn_counter_++;
                    if (final_turn_counter_ >= final_turn_frames_) {
                        final_turn_active_ = false;
                        state_ = State::DIGIT_SEARCH; 
                        ROS_INFO("Final turn done -> DIGIT_SEARCH");
                    }
                    break; 
                }
                if (final_forward_active_) {
                    linear_x = final_forward_linear_;
                    angular_z = final_forward_angular_;
                    final_forward_counter_++;
                    if (final_forward_counter_ >= final_forward_frames_) {
                        final_forward_active_ = false;
                        final_turn_active_ = true;
                        final_turn_counter_ = 0;
                        ROS_INFO("Final forward done -> Final Turn");
                    }
                    break;
                }

                int left_best_cx = -1, right_best_cx = -1;
                double left_best_area = 0.0, right_best_area = 0.0;
                for (const auto& obj : objects) {
                    if (obj.second < 500.0) continue;
                    if (obj.first < center_x) {
                        if (obj.second > left_best_area) { left_best_area = obj.second; left_best_cx = obj.first; }
                    } else {
                        if (obj.second > right_best_area) { right_best_area = obj.second; right_best_cx = obj.first; }
                    }
                }
                int lane_half_width_px = static_cast<int>(FRAME_WIDTH_ * 0.20);
                if (left_best_cx >= 0 && right_best_cx >= 0) lane_target_x = (left_best_cx + right_best_cx) / 2;
                else if (left_best_cx >= 0) lane_target_x = left_best_cx + lane_half_width_px;
                else if (right_best_cx >= 0) lane_target_x = right_best_cx - lane_half_width_px;

                linear_x = speed_normal_;
                if (lane_target_x >= 0) {
                    if (last_lane_target_x_ < 0) last_lane_target_x_ = lane_target_x;
                    double alpha = 0.5;
                    lane_target_x = static_cast<int>(alpha * lane_target_x + (1.0 - alpha) * last_lane_target_x_);
                    last_lane_target_x_ = lane_target_x;
                    int error = center_x - lane_target_x;
                    angular_z = KP_ * error;
                } else {
                    angular_z = 0.0;
                }

                if (!final_forward_active_ && !final_turn_active_) {
                    if (final_left_ratio < final_exit_ratio_max_ && final_right_ratio < final_exit_ratio_max_) {
                        final_forward_active_ = true;
                        final_forward_counter_ = 0;
                        final_forward_linear_ = linear_x;
                        final_forward_angular_ = angular_z;
                        ROS_INFO("Trigger Final Stage (Forward)");
                    }
                }
                break;
            }

            case State::DIGIT_SEARCH:
            {
                linear_x = 0.0;
                angular_z = digit_search_angular_; 
                
                double best_val = -1.0;
                std::string label = matchTemplate(gray_frame, best_val);
                
                if (best_val > 0.45) {
                    state_ = State::DIGIT_TRACK;
                    ROS_INFO("Digit Found: %s (Score: %.2f) -> DIGIT_TRACK", label.c_str(), best_val);
                    linear_x = 0.0;
                    angular_z = 0.0;
                }
                break;
            }

            case State::DIGIT_TRACK:
            {
                processTracking(gray_frame, linear_x, angular_z);
                break;
            }

            case State::FINISHED:
            {
                linear_x = 0.0;
                angular_z = 0.0;
                break;
            }
        }
    }

    std::string matchTemplate(const cv::Mat& gray, double& out_score) {
        double global_max = -1.0;
        std::string global_label = "";
        if (templates_.empty()) return "";

        for (auto const& item : templates_) {
            std::string label = item.first;
            std::vector<double> scales = {1.0, 0.75, 0.5, 0.4, 0.3};
            for (double scale : scales) {
                cv::Mat templ_scaled;
                if (scale == 1.0) templ_scaled = item.second;
                else cv::resize(item.second, templ_scaled, cv::Size(), scale, scale);
                
                if (gray.cols < templ_scaled.cols || gray.rows < templ_scaled.rows) continue;

                cv::Mat result;
                cv::matchTemplate(gray, templ_scaled, result, cv::TM_CCOEFF_NORMED);
                double minVal, maxVal;
                cv::minMaxLoc(result, &minVal, &maxVal, NULL, NULL);

                double score_penalty = 1.0;
                if (label == "1") score_penalty = 0.85; 
                double final_score = maxVal * score_penalty;

                if (final_score > global_max) {
                    global_max = final_score;
                    global_label = label;
                }
            }
        }
        out_score = global_max;
        return global_label;
    }

    void processTracking(const cv::Mat& gray, double& linear_x, double& angular_z) {
        double best_match_val = -1.0;
        cv::Rect best_match_rect;
        std::string best_match_label = "";

        if (!templates_.empty()) {
            for (auto const& item : templates_) {
                std::string label = item.first;
                cv::Mat templ = item.second;
                std::vector<double> scales = {1.0, 0.75, 0.5, 0.3};
                for (double scale : scales) {
                    cv::Mat templ_scaled;
                    if (scale == 1.0) templ_scaled = templ;
                    else cv::resize(templ, templ_scaled, cv::Size(), scale, scale);

                    if (gray.cols < templ_scaled.cols || gray.rows < templ_scaled.rows) continue;

                    cv::Mat result;
                    cv::matchTemplate(gray, templ_scaled, result, cv::TM_CCOEFF_NORMED);
                    double minVal, maxVal;
                    cv::Point minLoc, maxLoc;
                    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                    double score_penalty = 1.0;
                    if (label == "1") score_penalty = 0.9;
                    double final_score = maxVal * score_penalty;
                    double threshold = 0.55;

                    if (final_score > threshold && final_score > best_match_val) {
                        best_match_val = final_score;
                        best_match_label = label;
                        best_match_rect = cv::Rect(maxLoc.x, maxLoc.y, templ_scaled.cols, templ_scaled.rows);
                    }
                }
            }
        }

        best_digit_match_val_ = best_match_val;
        best_digit_label_ = best_match_label;
        if (best_match_val > 0) best_digit_rect_ = best_match_rect;

        if (!best_match_label.empty()) {
            cv::Mat roi_current = gray(best_match_rect);
            cv::Mat roi_thresh;
            cv::threshold(roi_current, roi_thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            int current_area = cv::countNonZero(roi_thresh);
            int target_ref_area = 25000;
            if (template_areas_.count(best_match_label)) target_ref_area = template_areas_[best_match_label];

            int center_x = gray.cols / 2;
            int target_cx = best_match_rect.x + best_match_rect.width / 2;
            int error_x = target_cx - center_x;
            int error_area = target_ref_area - current_area;

            angular_z = digit_track_kp_turn_ * error_x;
            double speed = digit_track_kp_dist_ * error_area;
            
            if (speed > 0.2) speed = 0.2;
            if (speed < -0.2) speed = -0.2;

            // Dead zone check for stopping, but stay in TRACKING state
            is_goal_reached_ = (std::abs(error_area) < target_ref_area * 0.1);
            if (is_goal_reached_) {
                speed = 0.0;
                angular_z = 0.0;
            } else {
                if (std::abs(angular_z) > 0.3) speed *= 0.5;
            }
            linear_x = speed;
        } else {
            linear_x = 0.0;
            angular_z = 0.0;
            is_goal_reached_ = false;
        }
    }

    void setSpeedParams(double speed_init, double speed_normal, double speed_turn,
                       double angular_turn_right, double angular_turn_left, double speed_stop, double kp) {
        speed_init_ = speed_init; speed_normal_ = speed_normal; speed_turn_ = speed_turn;
        angular_turn_right_ = angular_turn_right; angular_turn_left_ = angular_turn_left;
        speed_stop_ = speed_stop; KP_ = kp;
    }
    void setTurnFrames(int t1, int m, int t2) { turn1_frames_ = t1; move_frames_ = m; turn2_frames_ = t2; }
    void setGapThresholds(double el, double er, double xl, double xr) {
        enter_left_min_ = el; enter_right_min_ = er; exit_left_min_ = xl; exit_right_max_ = xr;
    }
    void setFinalStageParams(int ff, int ft, double fta, double fer) {
        final_forward_frames_ = ff; final_turn_frames_ = ft; final_turn_angular_ = fta; final_exit_ratio_max_ = fer;
    }
    void setGapTransitionParams(int p4m, int p5t) { phase4_move_frames_ = p4m; phase5_turn_frames_ = p5t; }
    void setDigitTrackParams(double kp_turn, double kp_dist) {
        digit_track_kp_turn_ = kp_turn;
        digit_track_kp_dist_ = kp_dist;
    }

    State getState() const { return state_; }
    bool isSelectingRoi() const { return selecting_roi_; }
    cv::Rect getRoiSelector() const { return roi_selector_; }
    cv::Rect getBestDigitRect() const { return best_digit_rect_; }
    std::string getBestDigitLabel() const { return best_digit_label_; }
    double getBestDigitVal() const { return best_digit_match_val_; }
    bool isGoalReached() const { return is_goal_reached_; }

private:
    State state_;
    int blind_spot_counter_, turn_counter_, alignment_counter_;
    bool align_before_turn_, final_forward_active_, final_turn_active_;
    int final_forward_counter_, final_turn_counter_, last_lane_target_x_;
    int SAFE_PASS_LIMIT_, MIN_TURN_DURATION_, FRAME_WIDTH_;
    double KP_, speed_init_, speed_normal_, speed_turn_;
    double angular_turn_right_, angular_turn_left_, speed_stop_;
    int turn1_frames_, move_frames_, turn2_frames_;
    int phase4_move_frames_, phase5_turn_frames_;
    double enter_left_min_, enter_right_min_, exit_left_min_, exit_right_max_;
    int final_forward_frames_, final_turn_frames_;
    double final_forward_linear_, final_forward_angular_, final_turn_angular_, final_exit_ratio_max_;

    double digit_search_angular_;
    double digit_track_kp_turn_, digit_track_kp_dist_;
    std::string template_path_;
    std::map<std::string, cv::Mat> templates_;
    std::map<std::string, int> template_areas_;
    
    bool selecting_roi_;
    cv::Point origin_;
    cv::Rect roi_selector_;
    bool is_goal_reached_;

    cv::Rect best_digit_rect_;
    std::string best_digit_label_;
    double best_digit_match_val_;
};

cv::Mat frame_msg;
RobotController controller;
ros::Publisher vel_pub;
cv::Scalar lower_red1(23, 125, 33);
cv::Scalar upper_red1(180, 255, 255);
cv::Mat kernel_noise = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
cv::Mat kernel_connect = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 35));

int roi_h = 200, roi_w = 200, roi_left_offset_x = 100, roi_right_offset_x = 100;
int final_roi_h = 300, final_roi_w = 400;

void clampRoiParams(int frame_width, int frame_height) {
    roi_h = std::max(1, std::min(roi_h, frame_height));
    roi_w = std::max(1, std::min(roi_w, frame_width));
    final_roi_h = std::max(1, std::min(final_roi_h, frame_height));
    final_roi_w = std::max(1, std::min(final_roi_w, frame_width));

    roi_left_offset_x = std::max(0, roi_left_offset_x);
    roi_right_offset_x = std::max(0, roi_right_offset_x);

    if (roi_left_offset_x + roi_w > frame_width) {
        roi_left_offset_x = std::max(0, frame_width - roi_w);
    }
    if (roi_right_offset_x + roi_w > frame_width) {
        roi_right_offset_x = std::max(0, frame_width - roi_w);
    }
    if (roi_left_offset_x + final_roi_w > frame_width) {
        roi_left_offset_x = std::max(0, frame_width - final_roi_w);
    }
    if (roi_right_offset_x + final_roi_w > frame_width) {
        roi_right_offset_x = std::max(0, frame_width - final_roi_w);
    }
}

void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
        frame_msg = cv_ptr->image.clone();
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void onMouseWrapper(int event, int x, int y, int flags, void* param) {
    if (frame_msg.empty()) return;
    cv::Mat gray;
    cv::cvtColor(frame_msg, gray, cv::COLOR_BGR2GRAY);
    controller.onMouse(event, x, y, flags, gray);
}

void drawVisualization(cv::Mat& display, const cv::Mat& mask_clean, 
                      int lane_target_x,
                      double left_ratio_small, double right_ratio_small,
                      double final_left_ratio, double final_right_ratio,
                      const std::vector<std::pair<int, double>>& objects,
                      double linear_x, double angular_z) {
    int width = display.cols;
    int height = display.rows;
    State state = controller.getState();
    
    cv::putText(display, "State: " + stateToString(state), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    if (state == State::DIGIT_SEARCH || state == State::DIGIT_TRACK || state == State::FINISHED) {
        if (state == State::DIGIT_SEARCH) {
             cv::putText(display, "Searching...", cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        } else if (state == State::DIGIT_TRACK) {
            std::string label = controller.getBestDigitLabel();
            cv::Rect rect = controller.getBestDigitRect();
            if (!label.empty()) {
                cv::rectangle(display, rect, cv::Scalar(0, 0, 255), 3);
                cv::putText(display, "Found: " + label, 
                       cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::putText(display, "Target Lost", cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
        } else if (state == State::FINISHED) {
            cv::putText(display, "GOAL REACHED", cv::Point(width/2 - 100, height/2), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
        }

        if (controller.isGoalReached()) {
             cv::putText(display, "GOAL REACHED", cv::Point(width/2 - 100, height/2), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
        }

        if (controller.isSelectingRoi()) {
            cv::rectangle(display, controller.getRoiSelector(), cv::Scalar(0, 255, 255), 2);
        }
        
    } else {
        cv::line(display, cv::Point(lane_target_x, 0), cv::Point(lane_target_x, height), cv::Scalar(0, 0, 255), 2);
        
        int curr_roi_h = (state == State::APPROACH_GAP) ? final_roi_h : roi_h;
        int curr_roi_w = (state == State::APPROACH_GAP) ? final_roi_w : roi_w;
        
        cv::rectangle(display, cv::Point(roi_left_offset_x, height - curr_roi_h), 
                      cv::Point(roi_left_offset_x + curr_roi_w, height), cv::Scalar(0, 255, 255), 2);
        cv::rectangle(display, cv::Point(width - curr_roi_w - roi_right_offset_x, height - curr_roi_h),
                      cv::Point(width - roi_right_offset_x, height), cv::Scalar(0, 255, 255), 2);
                      
        for (const auto& obj : objects) cv::circle(display, cv::Point(obj.first, height/2), 5, cv::Scalar(255, 0, 0), -1);
    }
}

void processFrame(const cv::Mat& frame) {
    if (frame.empty()) return;
    int width = frame.cols;
    int height = frame.rows;

    clampRoiParams(width, height);

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hsv, mask, mask_clean;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, lower_red1, upper_red1, mask);
    cv::morphologyEx(mask, mask_clean, cv::MORPH_OPEN, kernel_noise);
    cv::morphologyEx(mask_clean, mask_clean, cv::MORPH_CLOSE, kernel_connect);

    cv::Mat left_roi_small = mask_clean(cv::Rect(roi_left_offset_x, height - roi_h, roi_w, roi_h));
    cv::Mat right_roi_small = mask_clean(cv::Rect(width - roi_w - roi_right_offset_x, height - roi_h, roi_w, roi_h));
    double left_ratio_small = (double)cv::countNonZero(left_roi_small) / (roi_w * roi_h);
    double right_ratio_small = (double)cv::countNonZero(right_roi_small) / (roi_w * roi_h);

    int fr_h = std::min(final_roi_h, height);
    cv::Mat left_roi_final = mask_clean(cv::Rect(roi_left_offset_x, height - fr_h, final_roi_w, fr_h));
    cv::Mat right_roi_final = mask_clean(cv::Rect(width - final_roi_w - roi_right_offset_x, height - fr_h, final_roi_w, fr_h));
    double left_ratio_final = (double)cv::countNonZero(left_roi_final) / (final_roi_w * fr_h);
    double right_ratio_final = (double)cv::countNonZero(right_roi_final) / (final_roi_w * fr_h);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_clean, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::pair<int, double>> objects;
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area > 500) {
            cv::Moments M = cv::moments(cnt);
            if (M.m00 != 0) objects.push_back({static_cast<int>(M.m10 / M.m00), area});
        }
    }
    
    int lane_target_x = calculateLaneCenter(mask_clean);
    
    double linear_x, angular_z;
    controller.process(gray, objects,
                       left_ratio_small, right_ratio_small,
                       left_ratio_final, right_ratio_final,
                       lane_target_x, linear_x, angular_z);
    
    geometry_msgs::Twist vel_msg;
    vel_msg.linear.x = linear_x;
    vel_msg.angular.z = angular_z;
    vel_pub.publish(vel_msg);
    
    cv::Mat display = frame.clone();
    drawVisualization(display, mask_clean, lane_target_x,
                      left_ratio_small, right_ratio_small,
                      left_ratio_final, right_ratio_final,
                      objects, linear_x, angular_z);
    
    cv::imshow("Robot View", display);
    int key = cv::waitKey(1);
    if (key != -1) controller.handleDebugKeys(key, gray);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "exp_final_node");
    ros::NodeHandle n;
    ros::NodeHandle pn("~");

    double speed_init = 0.1, speed_normal = 0.2, speed_turn = 0.1, speed_stop = 0.0, kp = 0.002;
    double angular_turn_right = -0.3, angular_turn_left = 0.3;
    pn.param("speed_init", speed_init, speed_init);
    pn.param("speed_normal", speed_normal, speed_normal);
    pn.param("speed_turn", speed_turn, speed_turn);
    pn.param("angular_turn_right", angular_turn_right, angular_turn_right);
    pn.param("angular_turn_left", angular_turn_left, angular_turn_left);
    pn.param("speed_stop", speed_stop, speed_stop);
    pn.param("kp", kp, kp);

    pn.param("roi_h", roi_h, roi_h);
    pn.param("roi_w", roi_w, roi_w);
    pn.param("roi_left_offset_x", roi_left_offset_x, roi_left_offset_x);
    pn.param("roi_right_offset_x", roi_right_offset_x, roi_right_offset_x);
    pn.param("final_roi_h", final_roi_h, final_roi_h);
    pn.param("final_roi_w", final_roi_w, final_roi_w);

    int turn1_frames = 40, move_frames = 125, turn2_frames = 26;
    pn.param("turn1_frames", turn1_frames, turn1_frames);
    pn.param("move_frames", move_frames, move_frames);
    pn.param("turn2_frames", turn2_frames, turn2_frames);

    int phase4_move_frames = 20;
    int phase5_turn_frames = 20;
    pn.param("phase4_move_frames", phase4_move_frames, phase4_move_frames);
    pn.param("phase5_turn_frames", phase5_turn_frames, phase5_turn_frames);

    double enter_left_min = 0.5, enter_right_min = 0.3;
    double exit_left_min = 0.6, exit_right_max = 0.1;
    pn.param("enter_left_min", enter_left_min, enter_left_min);
    pn.param("enter_right_min", enter_right_min, enter_right_min);
    pn.param("exit_left_min", exit_left_min, exit_left_min);
    pn.param("exit_right_max", exit_right_max, exit_right_max);

    int final_forward_frames = 10, final_turn_frames = 30;
    double final_turn_angular = -0.3, final_exit_ratio_max = 0.02;
    pn.param("final_forward_frames", final_forward_frames, final_forward_frames);
    pn.param("final_turn_frames", final_turn_frames, final_turn_frames);
    pn.param("final_turn_angular", final_turn_angular, final_turn_angular);
    pn.param("final_exit_ratio_max", final_exit_ratio_max, final_exit_ratio_max);

    std::string template_path = "/home/eaibot/dip_ws/src/exp_final/templates/";
    pn.param<std::string>("template_path", template_path, template_path);

    double digit_kp_turn = -0.0025;
    double digit_kp_dist = 0.00001;
    pn.param("digit_kp_turn", digit_kp_turn, digit_kp_turn);
    pn.param("digit_kp_dist", digit_kp_dist, digit_kp_dist);

    controller.setSpeedParams(speed_init, speed_normal, speed_turn, angular_turn_right, angular_turn_left, speed_stop, kp);
    controller.setTurnFrames(turn1_frames, move_frames, turn2_frames);
    controller.setGapThresholds(enter_left_min, enter_right_min, exit_left_min, exit_right_max);
    controller.setFinalStageParams(final_forward_frames, final_turn_frames, final_turn_angular, final_exit_ratio_max);
    controller.setGapTransitionParams(phase4_move_frames, phase5_turn_frames);
    controller.setDigitTrackParams(digit_kp_turn, digit_kp_dist);
    controller.loadTemplates(template_path);

    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ros::Subscriber camera_sub = n.subscribe("/camera/color/image_raw", 1, rcvCameraCallBack);
    
    cv::namedWindow("Robot View", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Robot View", onMouseWrapper, NULL);
    
    ROS_INFO("Integrated Node Started. Default Template Path: %s", template_path.c_str());
    
    ros::Rate loop_rate(10);
    while (ros::ok()) {
        if (!frame_msg.empty()) processFrame(frame_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
    cv::destroyAllWindows();
    return 0;
}

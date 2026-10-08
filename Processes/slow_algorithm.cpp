#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm> // Required for std::min_element, std::max_element

struct DetectionResult {   // Sends to Fast Algorithm
    bool valid;
    float confidence;
    cv::Point2f corners[4];
    long long timestamp_ms;
};

struct TrackingStatusMsg {  // Recieves from Fast Algorithm
    bool request_detection;
    int active_points;
    float last_confidence;
    long long timestamp_ms;
};

int main() {
    const int width = 1080;
    const int height = 1920;
    const int channels = 3;
    const int frame_size = width * height * channels;

    zmq::context_t context(1);
    zmq::socket_t trigger_receiver(context, ZMQ_PULL);
    trigger_receiver.connect("tcp://127.0.0.1:5557");
    
    zmq::socket_t result_sender(context, ZMQ_PUSH);
    result_sender.connect("tcp://127.0.0.1:5558");

    int shm_fd = shm_open("/DroneFrameMemory", O_RDONLY, 0666);
    if (shm_fd == -1) {
        std::cerr << "Slow Algorithm: Failed to open shared memory." << std::endl;
        return -1;
    }
    void* shm_ptr = mmap(NULL, frame_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    
    sem_t* mutex_sem = sem_open("/DroneFrameMutex", 0);
    if (mutex_sem == SEM_FAILED) {
        std::cerr << "Slow Algorithm: Failed to open semaphore." << std::endl;
        return -1;
    }

    std::cout << "Slow Algorithm running. Awaiting triggers..." << std::endl;

    while (true) {
        zmq::message_t trigger_msg;
        (void)trigger_receiver.recv(trigger_msg, zmq::recv_flags::none);
        long long current_frame_timestamp = 0;
        if (trigger_msg.size() == sizeof(TrackingStatusMsg)) {
            TrackingStatusMsg* status = static_cast<TrackingStatusMsg*>(trigger_msg.data());
            current_frame_timestamp = status->timestamp_ms; 
            
            if (status->request_detection == true) {
                std::cout << "Tracking Lost: \n"
                          << "Last Confidence: " << status->last_confidence << std::endl;
            }
        }

        sem_wait(mutex_sem);
        cv::Mat shared_frame(height, width, CV_8UC3, shm_ptr);
        cv::Mat frame = shared_frame.clone(); // o.w keep the lock until the end of the process
        sem_post(mutex_sem);

        // detecting the A4 pad edges
        cv::Mat gray, blurred, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0); // remove noises for canny algo
        cv::Canny(blurred, edges, 50, 150); // 255 white, 0 black 
        std::vector<std::vector<cv::Point>> contours; 
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        DetectionResult result;
        result.valid = false;
        result.confidence = 0.0f;
        result.timestamp_ms = current_frame_timestamp;

        // prioritize evaluating the largest objects in the frame first
        std::sort(contours.begin(), contours.end(), 
                  [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                      return cv::contourArea(a) > cv::contourArea(b);
                  }); 

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt); 
            if (area < 5000) break; // A4 pad pixels above it. affected by altitude, resolution

            double epsilon = 0.02 * cv::arcLength(cnt, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cnt, approx, epsilon, true); // true = closed curve

            if (approx.size() == 4) {
                cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1); 
                std::vector<std::vector<cv::Point>> poly(1, approx);
                cv::fillPoly(mask, poly, cv::Scalar(255));  // fill the A4 white pad

                cv::Mat roi;
                gray.copyTo(roi, mask); // roi have the A4 pad with the X, all the other is black pixels
                cv::Mat thresh; 
                cv::adaptiveThreshold(roi, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

                double x_percentage_blackInk = (double)cv::countNonZero(thresh) / area; 
                if (x_percentage_blackInk > 0.02 && x_percentage_blackInk < 0.50) {
                    result.valid = true;
                    result.confidence = std::min(0.99, area / (width * height) + 0.5); // 0.5 bonus for correct closed A4 and ink 'X'
                    
                    // Radial sorting mechanism to correctly map 2D corners to 3D object points
                    cv::Point2f center(0, 0);
                    for (int i = 0; i < 4; ++i) {
                        center += static_cast<cv::Point2f>(approx[i]);
                    }
                    center.x /= 4.0f;
                    center.y /= 4.0f;
                    std::vector<cv::Point2f> temp_corners(4);
                    for (int i = 0; i < 4; ++i) {
                        temp_corners[i] = static_cast<cv::Point2f>(approx[i]);
                    }
                    // Sort corners clockwise based on their angle from the center point
                    std::sort(temp_corners.begin(), temp_corners.end(), [&center](const cv::Point2f& a, const cv::Point2f& b) {
                        return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
                    });
                    float top_edge = cv::norm(temp_corners[0] - temp_corners[1]); // of A4
                    float side_edge = cv::norm(temp_corners[1] - temp_corners[2]); // of A4
                    // 3. The 3D model expects a portrait orientation (side edge longer than top edge).
                    // If the top edge is longer in the 2D image, the paper is landscape.
                    // Shift the corners by 1 index to align the physical long edge with the 3D long edge.
                    if (top_edge > side_edge) {
                        result.corners[0] = temp_corners[1]; // TL
                        result.corners[1] = temp_corners[2]; // TR
                        result.corners[2] = temp_corners[3]; // BR
                        result.corners[3] = temp_corners[0]; // BL
                    } else {
                        for (int i = 0; i < 4; ++i) result.corners[i] = temp_corners[i];
                    }                         
                    break; 
                }
            }
        }
        std::cout << (result.timestamp_ms/1000.0) << " sec" << std::endl; 
        std::cout << std::endl;
        
        zmq::message_t result_msg(sizeof(DetectionResult));
        std::memcpy(result_msg.data(), &result, sizeof(DetectionResult));
        result_sender.send(result_msg, zmq::send_flags::none);
    }
    return 0;
}
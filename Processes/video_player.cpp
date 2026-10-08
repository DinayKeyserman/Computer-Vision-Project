#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <cstring>

struct OverlayData {  // recieves from Fast Algorithm
    bool valid;        // tracking data
    float distance;    // pose data
    float position[3]; // tvec pose data
    float orientation[3]; // pose data
    float confidence;     // overlay data  
    cv::Point2f corners[4];  // tracking data, 2D pixels image frame axis
};

cv::VideoCapture getVideoByIndex(int video_number) {
    const std::vector<std::string> video_paths = {
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Direct_Approach.mp4",
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Diagonal_Motion.mp4",
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Altitude_Change.mp4",
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Rotation_Motion.mp4",
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Side-to-side.mp4",
        "/mnt/c/Users/dinay/ComputerVisionAssignment/Videos/Distant_Pad_Exit_Entry.mp4"
    };
    return cv::VideoCapture(video_paths[video_number]);
}

cv::VideoCapture getValidVideoChoice() {
    int choice = 0; 
    while (true) {
        std::cout << "Please select a video (0-5): \n"
                  << "0 for Direct Approach\n"
                  << "1 for Diagonal Motion\n"
                  << "2 for Altitude Change\n"
                  << "3 for Rotation Motion\n"
                  << "4 for Side-to-side\n"
                  << "5 for Distant Pad Entry Exist\n";                
        if (std::cin >> choice && choice >= 0 && choice <= 5) {
            return getVideoByIndex(choice); 
        } 
        std::cout << "\nInvalid selection. You must enter a number between 0 and 5.\n" << std::endl;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

int main() {
    const int width = 1080;
    const int height = 1920;
    const int channels = 3; // colors: Red,Green,Blue. values 0-255
    const int frame_size = width * height * channels;

    // 1. Clean up previous crashed instances for safety
    shm_unlink("/DroneFrameMemory");
    sem_unlink("/DroneFrameMutex");

    // 2. Create POSIX Shared Memory
    int shm_fd = shm_open("/DroneFrameMemory", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory" << std::endl;
        return -1;
    }
    ftruncate(shm_fd,frame_size); // function changes size of fd
    void* shm_ptr = mmap(NULL, frame_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // 3. Create POSIX Named Semaphore (acting as a Mutex)
    sem_t* mutex_sem = sem_open("/DroneFrameMutex", O_CREAT, 0666, 1);
    if (mutex_sem == SEM_FAILED) {
        std::cerr << "Failed to create semaphore." << std::endl;
        return -1;
    }

    // 4. Setup ZeroMQ one-way communication channel
    zmq::context_t context(1); // 1 backround I/O thread for asynchronous communication
    zmq::socket_t frame_transmit(context, ZMQ_PUSH); // push can only send data, not recieve
    frame_transmit.bind("tcp://127.0.0.1:5555"); // bind means this process owns port 5555 waiting for fast-algo to connect
    
    zmq::socket_t overlay_rx(context, ZMQ_PULL); // pull is recieve data socket
    overlay_rx.bind("tcp://127.0.0.1:5556");

    cv::VideoCapture image_from_video = getValidVideoChoice();
    if (image_from_video.isOpened() == false) {
        std::cerr << "Failed to open the video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::cout << "Video Player running. Writing frames and awaiting overlay data..." << std::endl;
    std::cout << "Load fast and slow algorithms. The process begins after they loaded" << std::endl; 
    std::cin.get();

    while (image_from_video.read(frame)) {
        cv::rotate(frame, frame, cv::ROTATE_180);
        if (!frame.isContinuous()) frame = frame.clone();

        // 5. Lock and write directly to shared memory
        sem_wait(mutex_sem);
        std::memcpy(shm_ptr, frame.data, frame_size); // dest,src,amount of bytes
        sem_post(mutex_sem);

        long long current_timestamp = static_cast<long long>(image_from_video.get(cv::CAP_PROP_POS_MSEC));
        
        zmq::message_t trigger_msg(sizeof(long long));
        std::memcpy(trigger_msg.data(), &current_timestamp, sizeof(long long));
        frame_transmit.send(trigger_msg, zmq::send_flags::none);

        zmq::message_t overlay_msg;
        (void)overlay_rx.recv(overlay_msg, zmq::recv_flags::none); // The process stops at this line. It will wait until the Fast Algorithm process finishes its math and pushes the results back to the receiving socket on port 5556.
        OverlayData* overlay = static_cast<OverlayData*>(overlay_msg.data()); // match binary data to the struct fields

        // 6. Draw Overlay on Frame
        if (overlay->valid) {
            for (int i = 0; i < 4; ++i) {
                cv::line(frame, overlay->corners[i], overlay->corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 3); // draw green lines between each two corners
                cv::circle(frame, overlay->corners[i], 10, cv::Scalar(0, 0, 255), -1); // mark each corner with red dot
            }
            
            // 6.1 calculating the center of the pad
            cv::Point2f center(0, 0);
            for (int i = 0; i < 4; ++i) center += overlay->corners[i];
            center.x /= 4.0f; center.y /= 4.0f;
            cv::circle(frame, center, 8, cv::Scalar(255, 0, 0), -1); // blue point

            char textBuf[256];
            cv::putText(frame, "Status: Valid", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            sprintf(textBuf, "Distance: %.2f m", overlay->distance);
            cv::putText(frame, textBuf, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            sprintf(textBuf, "Position: [%.2f, %.2f, %.2f]", overlay->position[0], overlay->position[1], overlay->position[2]);
            cv::putText(frame, textBuf, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            sprintf(textBuf, "Orientation: [%.2f, %.2f, %.2f]", overlay->orientation[0], overlay->orientation[1], overlay->orientation[2]); 
            cv::putText(frame, textBuf, cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            sprintf(textBuf, "Confidence: %.2f", overlay->confidence);
            cv::putText(frame, textBuf, cv::Point(20, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        } else {
            cv::putText(frame, "Status: Invalid (Tracking Lost)", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("Drone Landing Pad Tracker", frame);
        int key = cv::waitKey(1);
        if (key == 'q') {
            break; 
        } 
        else if (key == ' ') { // Spacebar pressed
            std::cout << "Video Paused. Press Spacebar to resume." << std::endl;
            // cv::waitKey(0) waits infinitely until a key is pressed. 
            // loop because in case a key other than spacebar is pressed while paused.
            while (cv::waitKey(0) != ' ') {} 
            std::cout << "Video Resumed." << std::endl;
        }
    }

    // Cleanup POSIX resources
    munmap(shm_ptr, frame_size);
    close(shm_fd);
    sem_close(mutex_sem);
    shm_unlink("/DroneFrameMemory");
    sem_unlink("/DroneFrameMutex"); 
    return 0;
}
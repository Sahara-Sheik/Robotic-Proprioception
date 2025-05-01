"""
demonstration_recorder.py

Code that helps in recording ongoing demonstrations

"""
import sys
sys.path.append("..")



# from robotcontrol.gamepad_controller import GamepadController
# from robotcontrol.keyboard_controller import KeyboardController
# from robotcontrol.program_controller import ProgramController
from robot.al5d_position_controller import PositionController, RobotPosition
from camera.camera_controller import CameraController
import pathlib
import cv2
import json
import copy


class DemonstrationRecorder:
    """Record demonstration data collected from various controllers, sensors etc."""

    def __init__(self, controller, robot_controller: PositionController, camera_controller: CameraController, task_name = "unknown", save_dir = None):
        self.save_dir = save_dir
        self.remote_control = controller
        self.robot_controller = robot_controller
        self.camera_controller = camera_controller
        self.counter = 1
        self.task_name = task_name
        pass

    def save(self):
        """
        Write the data from the various sources with a common prefix
        """
        save_prefix = f"{self.counter:05d}"
        self.counter += 1
        data = {}
        data["rc-position-target"] = copy.copy(self.remote_control.pos_target.values)
        data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
        data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
        data["reward"] = 0.0 # placeholder for the reward
        data["annotation"] = "" # placeholder for annotation

        json_file = pathlib.Path(self.save_dir, f"{save_prefix}.json") 
        print(f"Saving into json file {json_file}")
        with open(json_file,"w") as f:
            json.dump(data, f)
        # save the captured images
        if self.camera_controller is not None:
            for index in self.camera_controller.images:
                filename = pathlib.Path(self.save_dir, f"{save_prefix}_{index}.jpg")
                cv2.imwrite(str(filename), self.camera_controller.images[index])

    def stop(self):
        """FIXME: This might be used in the future to save final stuff etc."""
        pass

    def convert_to_video(self):
        """Converts the jpg files to a video"""
        for index in self.camera_controller.images:        
            videopath = pathlib.Path(self.save_dir, f"video_{index}.mp4")

            # Collect and sort all files matching the pattern *_dev2.jpg
            image_files = sorted(self.save_dir.glob(f"*_{index}.jpg"))

            # Read the first image to get dimensions
            first_frame = cv2.imread(str(image_files[0]))
            height, width, _ = first_frame.shape
            fps = 10.0

            out = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            # Write all frames
            for image_file in image_files:
                frame = cv2.imread(str(image_file))
                out.write(frame)
            out.release()

    def access_frame(self, idx, frameno):
        """Accesses a particular frame in a particular video. 
        FIXME: this needs to be specified for the exact way in which this is used for training... it is probably a lot more expensive than just collecting the jpgs"""
        # Path to the video file
        video_path = pathlib.Path(self.save_dir, f"video_index.mp4")

        # Frame number you want to extract (e.g., 100th frame)
        target_frame = 100

        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Set the position to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameno)

        # Read the frame
        ret, frame = cap.read()

        #if ret:
            # Save the frame as an image
            # cv2.imwrite("frame_100.jpg", frame)
            #print("Frame saved as frame_100.jpg")
        #else:
        #    print("Failed to retrieve the frame.")

        cap.release()
        return frame
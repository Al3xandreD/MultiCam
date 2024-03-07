import time

from matplotlib import pyplot as plt


class StopError(Exception):
    def __init__(self, message="Attempt to stop measures but no measures have been started."):
        self.message = message
        super().__init__(self.message)


class ProfilingRecorder:

    def __init__(self):
        self.fps_list = []
        self.time_list = []
        self.total_fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        self.is_measuring = True

    def start(self):
        self.is_measuring = True
        self.start_time = time.time()

    def measure(self):
        if self.is_measuring:
            self.is_measuring = False
            end_time = time.time()
            delta_time = end_time - self.start_time
            fps = 1 / delta_time
            self.__record_benchmark(fps, delta_time)
        else:
            raise StopError("Attempt to stop measures but no measures have been started.")

    def __record_benchmark(self, fps, delta_time):
        self.total_fps += fps
        self.frame_count += 1
        self.fps_list.append(self.total_fps)
        self.time_list.append(delta_time)

    def plot_benchmark_res(self):
        avg_fps = self.total_fps / self.frame_count
        print(f"Average FPS: {avg_fps:.3f}")

        # plot the comparison graph
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('FPS')
        plt.title('FPS and Time Comparison Graph')
        plt.plot(self.time_list, self.fps_list, 'b', label="FPS & Time")
        #plt.savefig("FPS_and_Time_Comparison_pose_estimate.png")

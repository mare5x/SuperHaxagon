from queue import Empty
from multiprocessing import Process, Queue
import matplotlib.animation
from matplotlib import pyplot as plt


plot_queue = Queue()  # Plot tasks are put into this queue and taken by the plotting process.

def plot_loop(q):
    fig, ax = plt.subplots()

    def update(_):
        try:
            f, *args = q.get_nowait()
            # print(f, args)
            ax.cla()
            f(ax, *args)
        except Empty:
            pass
    
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=None)  # Docs: You must store the created Animation in a variable that lives as long as the animation should run. Otherwise, the Animation object will be garbage-collected and the animation stops.
    plt.show()

def start_plotting():
    proc = Process(target=plot_loop, args=(plot_queue,), daemon=True)
    proc.start()
    print(f"Plotter PID: {proc.pid}")

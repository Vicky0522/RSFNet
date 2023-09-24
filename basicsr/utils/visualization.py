#########################
## visualizer
#########################

from .visualizer import Visualizer as vs

class Visualizer(object):
    def __init__(self, \
            print_freq = 10, \
            display_id=1, \
            display_port=8109, \
            display_winsize=1024, \
            name='train', \
            display_ncols=0, \
            display_server="http://localhost", \
            display_env='main',
            check_points="./checkpoints"
    ):
        from collections import namedtuple
        Opt = namedtuple("Opt", "print_freq display_id display_port display_winsize name display_ncols display_server display_env check_points")
        v_opt = Opt(print_freq = print_freq, \
                    display_id = display_id, \
                    display_port = display_port, \
                    display_winsize = display_winsize, \
                    name = name, \
                    display_ncols = display_ncols, \
                    display_server = display_server, \
                    display_env = display_env, \
                    check_points = check_points \
                   )
        self.visualizer = vs(v_opt)

    def display_current_results(self, visuals, epoch, save_result):
        self.visualizer.display_current_results(visuals, epoch, save_result)

"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
import pprint as pp
from collections import namedtuple

from tensorboard.backend.event_processing import event_accumulator
#
# ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])
# CompressedHistogramEvent = namedtuple('CompressedHistogramEvent',
#                                       ['wall_time', 'step',
#                                        'compressed_histogram_values'])
# HistogramEvent = namedtuple('HistogramEvent',
#                             ['wall_time', 'step', 'histogram_value'])
# HistogramValue = namedtuple('HistogramValue', ['min', 'max', 'num', 'sum',
#                                                'sum_squares', 'bucket_limit',
#                                                'bucket'])
# ImageEvent = namedtuple('ImageEvent', ['wall_time', 'step',
#                                        'encoded_image_string', 'width',
#                                        'height'])
# AudioEvent = namedtuple('AudioEvent', ['wall_time', 'step',
#                                        'encoded_audio_string', 'content_type',
#                                        'sample_rate', 'length_frames'])
# TensorEvent = namedtuple('TensorEvent', ['wall_time', 'step', 'tensor_proto'])


class Plotter:
    """Extract image/scalars/... from tensorboard files and save it to images"""

    def __init__(self, plot_dir):
        self.plot_dir = plot_dir

    def dynamic_board(self, file, mode=None):
        if mode is None or mode == "created":
            self.ea = event_accumulator.EventAccumulator(file,
                                                         size_guidance={  # see below regarding this argument
                                                             event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                             event_accumulator.IMAGES: 4,
                                                             event_accumulator.AUDIO: 4,
                                                             event_accumulator.SCALARS: 0,
                                                             event_accumulator.HISTOGRAMS: 1,
                                                         })
            print("\nLoad board in {}".format(file))

    def get_scalars(self, tag):
        return self.ea.Scalars(tag)  # 'wall_time', 'step', 'value'

    def get_histograms(self, tag):
        return self.ea.Histograms(tag)

    def get_audio(self, tag):
        return self.ea.Audio(tag)

    def get_images(self, tag):
        return self.ea.Images(tag)

    def get_compressed_histograms(self, tag):
        return self.ea.CompressedHistograms(tag)

    def get_tensor(self, tag):
        return self.ea.Tensors(tag)

    def summary(self):
        self.ea.Reload()
        pp.pprint(self.ea.Tags())

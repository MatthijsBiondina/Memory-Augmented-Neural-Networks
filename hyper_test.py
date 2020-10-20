import subprocess

from utils.hyperparameters import Hyper
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")
#
# # Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)
#
#         plt.show()



H = Hyper()
H.plot_scores('learning_rate')
H.plot_scores('batch_size')
H.heatmaps()

H.merge()

#
# # plt.savefig('./res/results.pdf')
#
# plt.show()
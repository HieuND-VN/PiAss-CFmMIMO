import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

trloss_MLP = np.array([10.9393502 ,  6.7209339 ,  4.14170271,  3.54486408,  3.24090282,  3.02047289, 2.91489279,  2.85181662,  2.57222434,  2.53253645,
                       2.51574442,  2.47869929,   2.45739625,  2.5173593 ,  2.41615197,  2.3427371 , 2.32337204,  2.31180838,  2.28995883,  2.27432818,
                       2.26894884,  2.26308358,   2.26079073,  2.25595523,  2.25353031,  2.24937643, 2.24716456,  2.24408476,  2.2379087 ,  2.23242743,
                       2.22663648,  2.22396621,   2.22177081,  2.32874176,  2.3269665 ,  2.325789,   2.23378514,  2.28145241,  2.30243662,  2.33767501,
                       2.41996698,  2.35736521,   2.37799125,  2.31024528,  2.37667816,  2.33787943, 2.35328866,  2.33787943,  2.32663648,  2.32308358, 2.35328866])

trloss_HQCNN = np.array([4.32442156, 3.50861664, 3.0294157,  2.7018044 , 2.50101199, 2.40038055, 2.30055441, 2.30038431, 2.30039311, 2.30043735,
                         2.30042373, 2.30043364, 2.30046635, 2.3003341 , 2.30029965, 2.30037063, 2.30043666, 2.30036823, 2.30035896, 2.30031766,
                         2.30039315, 2.30037159, 2.30031281, 2.3003708 , 2.30031743, 2.30030543, 2.30018966, 2.30034909, 2.30013502, 2.30032539,
                         2.30030894, 2.30017496, 2.30036174, 2.30041557, 2.30038527, 2.30038371, 2.30039355, 2.3003348 , 2.30045897, 2.30032288,
                         2.3003665 , 2.30024791, 2.30029408, 2.30031681, 2.3003055 , 2.30032645, 2.30022242, 2.30045907, 2.30025882, 2.30024433, 2.30022242])

trloss_CNN = np.array([6.49331224, 4.20230875, 3.10080528, 2.7639718,  2.73032376, 2.63031595, 2.63031789, 2.62032005, 2.55032666, 2.56035014,
                       2.50035611, 2.48035811, 2.4435977,  2.45836121, 2.46036245, 2.47036364, 2.46036458, 2.45036542, 2.45036625, 2.4303669 ,
                       2.44036762, 2.41036814, 2.42036869, 2.40036914, 2.33936964, 2.33537004, 2.33837044, 2.33637083, 2.33137118, 2.3337149,
                       2.30037176, 2.30037203, 2.30037227, 2.30037251, 2.30037273, 2.30037299, 2.32037327, 2.32037338, 2.3133736 , 2.32437375,
                       2.32837396, 2.3103741,  2.30037423, 2.30037446, 2.31037454, 2.32037471, 2.33037482, 2.30037498, 2.29037505, 2.31037517,2.33037482])
# plt.figure(figsize=(10, 6))778

plt.plot(trloss_MLP, linestyle='--', label='MLP')
plt.plot(trloss_CNN, linestyle='-.', label='CNN')
plt.plot(trloss_HQCNN, label='HQCNN')
plt.legend(fontsize=14)
plt.xlim(-1, 50)
plt.ylim(2, 6)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss value', fontsize=16)
plt.grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add rectangle to highlight the zoomed region
rect = Rectangle((44, 2.2), 5, 0.4, linewidth=1, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)

# Zoomed-in region
axins = zoomed_inset_axes(plt.gca(), zoom=4, loc='center right', bbox_to_anchor=(1, 0.35), bbox_transform=plt.gca().transAxes)
axins.plot(trloss_MLP, linestyle='--', label='MLP')
axins.plot(trloss_CNN, linestyle='-.', label='CNN')
axins.plot(trloss_HQCNN, label='HQCNN')
axins.set_xlim(44, 49)
axins.set_ylim(2.25, 2.45)
axins.grid(True)

# Remove ticks in the inset
axins.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axins.tick_params(axis='y', which='major', labelsize=8)

# Connect the rectangle and inset
mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")

# Add legend at the upper middle of the main plot


# Save and display the figure
plt.savefig('supervised_loss_clean_zoom.png', bbox_inches='tight', dpi=600)
plt.show()
import numpy as np


class BetaRadomization():

    def __init__(self, beta, seed):
        """
        Do initliatization
        """


        self.mhf = 2 # maximal horzontal frequency
        self.mvf = 5 # maximal vertical frequency
        self.height_max = 5
        self.offset = []


        self.beta = beta

        # # sample number of furier components, sample random offsets to one another, # Independence Height and angle
        # self.number_height = np.random.randint(3,5)
        # self.number_angle = np.random.randint(6,10)

        # # sample frequencies
        # self.frequencies_angle = np.random.randint(1, self.mhf, size=self.number_angle)
        # self.frequencies_height = np.random.randint(0, self.mvf, size=self.number_angle)
        # # sample frequencies
        # self.offseta = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        # self.offseth = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        # self.intensitya = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)
        # self.intensityh = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)

        
        # Initialize rng
        self.rng = np.random.default_rng(seed=seed)

        # sample number of furier components, sample random offsets to one another, # Independence Height and angle
        self.number_height = self.rng.integers(3, 5)
        self.number_angle = self.rng.integers(6,10)

        # sample frequencies
        self.frequencies_angle = self.rng.integers(1, self.mhf, size=self.number_angle)
        self.frequencies_height = self.rng.integers(0, self.mvf, size=self.number_angle)
        # sample frequencies
        self.offseta = self.rng.uniform(0, 2*np.pi, size=self.number_angle)
        self.offseth = self.rng.uniform(0, 2*np.pi, size=self.number_angle)
        self.intensitya = self.rng.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)
        self.intensityh = self.rng.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)


    def propagate_in_time(self, timestep):
        self.offseta += self.frequencies_angle * timestep/10
        self.offseth += self.frequencies_height * timestep / 10
        pass

    def setup(self, beta):
        pass

    def _function(self, angle_h=None, height=None):
        was_None = False
        if height is None:
            height = np.linspace(0, self.height_max, 200)/self.height_max*2*np.pi
            was_None = True

        if angle_h is None:
            angle_h = np.linspace(0, 2*np.pi, 200)
            was_None = True
        a = 0
        h = 0
        if was_None:
            a, h = np.meshgrid(angle_h, height)
        else:
            a = angle_h
            h = height

        output = np.zeros(np.shape(a))
        for fa, fh, oa, oh, Ah, Aa in zip(self.frequencies_angle, self.frequencies_height, self.offseta, self.offseth, self.intensityh, self.intensitya):
            output += np.abs((Aa*np.sin(fa*a+oa)/fa+Ah*np.sin(fa*a+fh*h+oh)))

        output += self.beta
        # print(output)
        return output

    def _print_function(self):
        """
        Print function for values inbetween 0-360 and inbetween different heights
        :return:
        """
        pass




    def get_beta(self, distance_forward, right, height):
        distance_forward = np.where(distance_forward == 0, np.ones_like(distance_forward) * 0.0001, distance_forward)
        angle = np.tan(np.divide(right, distance_forward))
        beta_usefull = self._function(angle, height)

        return beta_usefull


def haze_point_cloud(pts_3D, random_beta, fraction_random, seed):
    #print 'minmax_values', max(pts_3D[:, 0]), max(pts_3D[:, 1]), min(pts_3D[:, 1]), max(pts_3D[:, 2]), min(pts_3D[:, 2])
    # n = []
    # foggyfication should be applied to sequences to ensure time correlation inbetween frames
    # vectorze calculation
    # print pts_3D.shape
    # if args.sensor_type=='VelodyneHDLS3D':
    #     # Velodyne HDLS643D
    #     n = 0.04
    #     g = 0.45
    #     dmin = 2 # Minimal detectable distance
    # elif args.sensor_type=='VelodyneHDLS2':
    #     #Velodyne HDL64S2
    #     n = 0.05
    #     g = 0.35
    #     dmin = 2
    rng = np.random.default_rng(seed)

    # Velodyne HDLS643D
    n = 0.04
    g = 0.45
    dmin = 2 # Minimal detectable distance

    d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
    detectable_points = np.where(d>dmin)
    d = d[detectable_points]
    pts_3D = pts_3D[detectable_points]

    beta_usefull = random_beta.get_beta(pts_3D[:,0], pts_3D[:, 1], pts_3D[:, 2])
    dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
    dnew = -np.log(1 - 0.5) / (beta_usefull)

    probability_lost = 1 - np.exp(-beta_usefull*dmax) #Prob lost for each shape
    #lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost
    lost = rng.uniform(0, 1, size=probability_lost.shape) < probability_lost

    lost_action_prob = np.prod([p**lost[i] * (1-p)**lost[i] for i,p in enumerate(probability_lost)])


    if random_beta.beta == 0.0:
        dist_pts_3d = np.zeros((pts_3D.shape[0], 6))
        dist_pts_3d[:, 0:5] = pts_3D
        dist_pts_3d[:, 5] = np.zeros(np.shape(pts_3D[:, 3]))
        return dist_pts_3d,  []

    cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
    idx_stable = np.where(d<dmax)[0]
    old_points = np.zeros((len(idx_stable), 6))
    old_points[:,0:5] = pts_3D[idx_stable,:]
    old_points[:, 3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
    old_points[:, 5] = np.zeros(np.shape(old_points[:,3]))

    cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
    cloud_scatter = np.zeros((len(cloud_scatter_idx), 6))
    cloud_scatter[:,0:5] =  pts_3D[cloud_scatter_idx,:]
    cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
    cloud_scatter[:, 3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
    cloud_scatter[:, 5] = np.ones(np.shape(cloud_scatter[:, 3]))


    # Subsample random scatter abhaengig vom noise im Lidar
    random_scatter_idx = np.where(random_scatter)[0]
    scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
    #drand = np.random.uniform(high=scatter_max[random_scatter_idx])
    drand = rng.uniform(high=scatter_max[random_scatter_idx])
    # scatter outside min detection range and do some subsampling. Not all points are randomly scattered.
    # Fraction of 0.05 is found empirically.
    drand_idx = np.where(drand>dmin)
    drand = drand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]
    # Subsample random scattered points to 0.05%
    # print(len(random_scatter_idx), fraction_random)
    
    # subsampled_idx = np.random.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
    subsampled_idx = rng.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
    drand = drand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]


    random_scatter = np.zeros((len(random_scatter_idx), 6))
    random_scatter[:,0:5] = pts_3D[random_scatter_idx,:]
    random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
    random_scatter[:, 3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
    random_scatter[:, 5] = 2*np.ones(np.shape(random_scatter[:, 3]))



    dist_pts_3d = np.concatenate((old_points, cloud_scatter,random_scatter), axis=0)

    color = []
    return dist_pts_3d, lost_action_prob
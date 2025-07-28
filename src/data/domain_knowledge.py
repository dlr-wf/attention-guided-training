import numpy as np
from scipy.interpolate import griddata
from crackpy.structure_elements.material import Material
from crackpy.fracture_analysis.optimization import Optimization
from crackpy.fracture_analysis.crack_tip import williams_displ_field, williams_stress_field

class WilliamsDomainKnowledge():

    def __init__(self, 
                 lower_threshold : float = 100, 
                 upper_threshold : float = 200, 
                 type="gradual williams"):
        """
        Initialize the Williams domain knowledge class with the given parameters.
        Args:
            lower_threshold (float): Lower threshold for the explanation preparation.
            upper_threshold (float): Upper threshold for the explanation preparation.
            type (str): Type of explanation to generate. Options are "binary williams", "gradual williams", "binary misleading", "multi-gradual misleading".
        
        Raises:
            ValueError: If the 'type' is not recognized.
        """
        self.lower_thresh = lower_threshold
        self.upper_thresh = upper_threshold
        self.expl_type = type.lower()
        
        self.new_sigma_vm = self.generate_williams_field() # generate the representative williams field once for future modification
        
    def __call__(self,
                 x : float,
                 y : float) -> np.ndarray:
        """
        Retreive Williams field and shift it to the ground truth crack position, or generate missleading/fake explanations.
        Applies thresholds and min-max scaling for gradient and binary explanations.
        
        Args:       
            x (float): positions of the tips
            y (float): positions of the tips
        
        Returns:
            np.ndarray: Explanation map based on the type specified during initialization.
        """
        
        if self.expl_type == "binary williams":
            tip_field = self.get_tip_field(x,y)
            return np.where(tip_field > (self.upper_thresh + self.lower_thresh)/2, 1, 0)
        
        elif self.expl_type == "gradual williams":
            tip_field = self.get_tip_field(x,y)

            map = np.where(tip_field > self.lower_thresh, tip_field, self.lower_thresh)
            map = np.where(map < self.upper_thresh, map, self.upper_thresh)
            map = (map - self.lower_thresh)/(self.upper_thresh - self.lower_thresh)
            return map
        
        elif self.expl_type == "binary misleading":
            map = np.zeros((256,256))
            map[180:,180:] = 1
            return map 

        elif self.expl_type == "multi-gradual misleading":
            shape=(256, 256)
            radius=70
            max_value=1
            decay_rate=1.3e-2
            
            def create_circle(center, radius, max_value, decay_rate):
                y, x = np.ogrid[-center[0]:shape[0]-center[0], -center[1]:shape[1]-center[1]]
                dist_from_center = np.sqrt(x*x + y*y)
                mask = dist_from_center <= radius
                circle = np.zeros(shape)
                circle[mask] = max_value * np.exp(-decay_rate * dist_from_center[mask])
                return circle

            map = np.zeros(shape)
            lower_right = create_circle((shape[0], shape[1]), radius, max_value, decay_rate)
            upper_right = create_circle((0, shape[1]), radius, max_value, decay_rate)
            
            map = np.maximum(map, lower_right)
            map = np.maximum(map, upper_right)
            
            return map
        else:
            raise ValueError(f"Explanation type '{self.expl_type}' is not recognized. Choose from 'binary williams', 'gradual williams', 'binary misleading', or 'multi-gradual misleading'.")

    def get_tip_field(self,
                      x : float,
                      y : float,
                      size : int = 256):
        """
        Get the Williams field and shift it to the correct position of the tip.
        
        Args:
            x (float): x-coordinate of the crack tip.
            y (float): y-coordinate of the crack tip.
            size (int): Size of the output field. Default is 256.
            
        Returns:
            np.ndarray: The shifted Williams field centered at the specified crack tip position.
        """
        center = int(size/2)
        offset_tip_from_center = (-int(x-center),-int(y-center))
        return self.new_sigma_vm[(center+offset_tip_from_center[0]):(-center+offset_tip_from_center[0]),
                                 (center+offset_tip_from_center[1]):(-center+offset_tip_from_center[1])]
    
    def ReturnWilliamsMaps(self,
                           K_I : float = 23.7082070388 * np.sqrt(1000),
                           T : float = -7.2178311164) -> np.ndarray:
        """
        Return Williams maps for the given material properties, StressIntensityFactor and T-stress.
        
        Args:
            K_I (float): Stress intensity factor.
            T (float): T-stress value.
        
        Returns:
            np.ndarray: The von Mises stress field, x-coordinates, and y-coordinates of the Williams field.
        """
        # Parameters based on n20.00_20.00_0.00_nodemap_right_Output.txt <-> representative FE simulation
        A_1 = K_I / np.sqrt(2 * np.pi)
        A_2 = T / 4
        A_3 = -2.8672068762
        A_4 = 0.0290671612

        A = [A_1, A_2, A_3, A_4]
        B = [0, 0, 0, 0] # Assume pure Mode I loading, so B
        terms = [1,2,3,4] # -> a_i, b_i for i in terms

        min_radius = 0.01
        max_radius = 30.0
        r_grid, phi_grid = np.mgrid[min_radius:max_radius:((max_radius-min_radius)/256),
                                    -np.pi:np.pi:(2*np.pi/256)]
        
        sigma_xx, sigma_yy, sigma_xy = williams_stress_field(A, B, terms, phi_grid, r_grid)
        sigma_vm = np.sqrt(sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * sigma_xy ** 2) # von Mises stress field
        
        x_grid, y_grid = Optimization.make_cartesian(r_grid, phi_grid)
        return sigma_vm, x_grid, y_grid
    
    def generate_williams_field(self):
        """
        Generate Williams fields based on Crackpy implementation.
        """
        # get basic williams field
        sigma_vm, x_grid, y_grid = self.ReturnWilliamsMaps()
        x_min, x_max = np.min(x_grid), np.max(x_grid)
        y_min, y_max = np.min(y_grid), np.max(y_grid)

        x_pts = 512
        y_pts = 512

        new_x = np.linspace(x_min, x_max, x_pts)
        new_y = np.linspace(y_min, y_max, y_pts)
        new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)

        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        values = sigma_vm.ravel()
        new_points = np.column_stack((new_x_grid.ravel(), new_y_grid.ravel()))

        new_sigma_vm = griddata(points, values, new_points, method="linear") 
        return np.nan_to_num(new_sigma_vm.reshape(new_x_grid.shape),0)

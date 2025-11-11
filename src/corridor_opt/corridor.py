"""
Use cases:
c = Corridor(p0, pf, r, alpha, width)
energy_est = EnergyEstimator()
energy_est.get(corridor, weather_sample)
traffic_density_est.get(corridor, ais_sample)
"""

from corridor_opt.obstacle import Obstacle
from typing import List, Tuple, Optional
import numpy as np
from shapely import LineString
from corridor_opt.corridor_utils import get_rectangle_and_bend_from_wpts
from colorama import Fore, Style, init




class Corridor(Obstacle):
    backbone: LineString # Will be useful to integrate power along corridor, compute distance w.r.t corridor

    def __init__(
            self,
            wp1: Tuple[float, float], # Anchor point
            wp2: Tuple[float, float], # progression point
            dir1: Tuple[float, float],
            radius: float,
            angle: float,
            width: float,
            prev_main_node: Optional[int] = None,   # Previous main node (i.e. starting point if it is the first corridor of the edge)
            next_main_node: Optional[int] = None,   # Next main node (i.e. final point if it is the last corridor of the edge)
            idx: Optional[int] = None,              # Order among all the sub-corridors along the edge -> \in [0, \infty[
            length_margin: float = 0,
            n_radius_approx: int = 10,
            degrees: bool = False,
            edge_id: Optional[int] = None,
    ):

        self.wp1 = wp1
        self.wp2 = wp2
        self.dir1 = dir1
        self.radius = radius
        self.angle = np.deg2rad(angle) if degrees else angle
        self.width = width
        self.prev_main_node = prev_main_node
        self.next_main_node = next_main_node
        self.idx = idx
        self.length_margin = length_margin
        self.n_radius_approx = n_radius_approx
        self.edge_id = edge_id
        

        xy = self.init_corridor()
        super().__init__(xy=xy)

    def init_corridor(self) -> List[Tuple[float, float]]:
        """
        Returns the x,y coordinates of the corridor's shape based on its parameterization.
        """
        rect, corridor, backbone = get_rectangle_and_bend_from_wpts(
                wp1=self.wp1,
                wp2=self.wp2,
                dir1=self.dir1,
                radius=self.radius,
                angle=self.angle,
                width=self.width,
                length_margin=self.length_margin,
                n_radius_approx=self.n_radius_approx
            )
        self.rect = rect
        self.backbone = backbone
        return corridor.get_xy_as_list()
    
    def signed_distance(self, x: float, y: float) -> float:
        """
        Compute the signed distance of a point x, y to the corridor, i.e. if the point is within the corridor, distance < 0.
        Useful to compute traffic density
        """
        return self.distance_to_obstacle(x, y)
    
    def average_orientation(self, n_samples: int = 20) -> float:
        """
        Returns the average orientation of the backbone using n_samples evenly spaced.
        """
        psi = 0
        for p in np.linspace(0, 1, n_samples):
            psi += self.orientation(p, normalized=True) / n_samples
        return psi

    def orientation(self, progression: float, normalized: bool = False) -> float:
        """
        Returns the orientation of the backbone at a given progression value. Useful for integrating power consumption along corridor.
        """
        length = 1 if normalized else self.backbone.length
        assert 0 <= progression <= length , f"progression must be within [0, {length:.1f}]. Got progression={progression:.2f}"
        
        if progression == 0:
            prog_prev = 0
            prog_next = 0.005 * length
        else: # meaning 0 < progression <= length
            prog_prev = 0.995 * progression
            prog_next = progression

        p1 = self.backbone_coord(prog_prev, normalized=normalized)
        p2 = self.backbone_coord(prog_next, normalized=normalized)

        return np.atan2(p2[0]-p1[0], p2[1]-p1[1])
    
    def backbone_coord(self, progression: float, normalized: bool = False) -> Tuple[float, float]:
        p = self.backbone.interpolate(progression, normalized=normalized)
        return (p.x, p.y)
    
    def export(self, folder: Optional[str] = None, filename: Optional[str] = None) -> bool:
        import pathlib, os
        from datetime import datetime
        
        if folder is None:
            folder = str(pathlib.Path(__file__).parent)
        if filename is None:
            # Add timestamp (hours, minutes, seconds) to filename
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"corridor_between_{self.prev_main_node}_{self.next_main_node}_idx_{self.idx}_{timestamp}.txt"
        path = os.path.join(folder, filename)

        try:
            with open(path, 'x') as f:
                # Write header
                f.write("# Corridor parameters for get_rectangle_and_bend_from_wpts\n")
                f.write("# Format: parameter_name=value\n")
                f.write("\n")
                
                # Write all required parameters for get_rectangle_and_bend_from_wpts
                f.write(f"[wp1] {self.wp1[0]},{self.wp1[1]}\n")
                f.write(f"[wp2] {self.wp2[0]},{self.wp2[1]}\n")
                f.write(f"[dir1] {self.dir1[0]},{self.dir1[1]}\n")
                f.write(f"[radius] {self.radius}\n")
                f.write(f"[angle] {self.angle}\n")  # stored in radians
                f.write(f"[width] {self.width}\n")
                f.write(f"[length_margin] {self.length_margin}\n")
                f.write(f"[n_radius_approx] {self.n_radius_approx}\n")
                
                # Write additional corridor metadata
                f.write(f"[prev_main_node] {self.prev_main_node}\n")
                f.write(f"[next_main_node] {self.next_main_node}\n")
                f.write(f"[idx] {self.idx}\n")
                f.write(f"[edge_id] {self.edge_id}\n")
                
            return True
            
        except FileExistsError:
            print(f"File {path} already exists. Use a different path or delete the existing file.")
            return False
        except Exception as e:
            print(f"Error writing to file {path}: {e}")
            return False
        
    @staticmethod
    def load(path: str) -> "Corridor":
        """
        Load a Corridor from a text file exported by the export method.
        
        Args:
            path: Path to the exported corridor file
            
        Returns:
            Corridor: A new Corridor instance with the loaded parameters
        """
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corridor file not found: {path}")
        
        # Dictionary to store parsed parameters
        params = {}
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse lines in format: [parameter_name] value
                    if line.startswith('[') and ']' in line:
                        # Extract parameter name and value
                        bracket_end = line.index(']')
                        param_name = line[1:bracket_end]
                        value_str = line[bracket_end + 1:].strip()
                        
                        # Parse different parameter types
                        if param_name in ['wp1', 'wp2', 'dir1']:
                            # Parse tuple coordinates: "x,y"
                            x_str, y_str = value_str.split(',')
                            params[param_name] = (float(x_str), float(y_str))
                        elif param_name in ['radius', 'angle', 'width', 'length_margin']:
                            # Parse float values
                            params[param_name] = float(value_str)
                        elif param_name in ['n_radius_approx', 'prev_main_node', 'next_main_node', 'idx', 'edge_id']:
                            # Parse integer values (handle None for optional fields)
                            if value_str.lower() == 'none':
                                params[param_name] = None
                            else:
                                params[param_name] = int(value_str)
            
            # Create and return Corridor instance with loaded parameters
            return Corridor(
                wp1=params['wp1'],
                wp2=params['wp2'],
                dir1=params['dir1'],
                radius=params['radius'],
                angle=params['angle'],  # already in radians from file
                width=params['width'],
                prev_main_node=params.get('prev_main_node'),
                next_main_node=params.get('next_main_node'),
                idx=params.get('idx'),
                length_margin=params.get('length_margin', 0),
                n_radius_approx=params.get('n_radius_approx', 10),
                degrees=False,  # angle is already in radians from file
                edge_id=params.get('edge_id')
            )
            
        except Exception as e:
            raise ValueError(f"Error parsing corridor file {path}: {e}")
        
    @staticmethod
    def load_all_corridors_in_folder(folder_path: str) -> List["Corridor"]:
        """
        Load all corridor files from a folder and return them as a list of Corridor objects.
        
        Args:
            folder_path: Path to the folder containing corridor .txt files
            
        Returns:
            List[Corridor]: List of loaded Corridor instances
            
        Raises:
            FileNotFoundError: If the folder doesn't exist
            ValueError: If no valid corridor files are found or parsing errors occur
        """
        import os
        import glob
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        corridors = []
        failed_files = []
        
        # Find all .txt files in the folder
        txt_pattern = os.path.join(folder_path, "*.txt")
        txt_files = glob.glob(txt_pattern)
        
        if not txt_files:
            raise ValueError(f"No .txt files found in folder: {folder_path}")
        
        # Initialize colorama
        init(autoreset=True)
        
        # Attempt to load each file as a corridor
        for file_path in txt_files:
            try:
                # Check if this is likely a corridor file by looking for expected parameters
                if Corridor._is_corridor_file(file_path):
                    corridor = Corridor.load(file_path)
                    corridors.append(corridor)
                    print(f"{Fore.GREEN} Successfully loaded corridor from: {os.path.basename(file_path)}{Style.RESET_ALL}")
                else:
                    print(f" Skipping non-corridor file: {os.path.basename(file_path)}")
                    
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"{Fore.RED} Failed to load {os.path.basename(file_path)}: {e}{Style.RESET_ALL}")
        
        if failed_files and not corridors:
            # All files failed to load
            error_msg = "Failed to load any corridors. Errors:\n"
            for file_path, error in failed_files:
                error_msg += f"  - {os.path.basename(file_path)}: {error}\n"
            raise ValueError(error_msg)
        
        if failed_files:
            print(f"{Fore.YELLOW} Warning: {len(failed_files)} files failed to load, {len(corridors)} corridors loaded successfully.{Style.RESET_ALL}")
        
        # Sort corridors by idx if available for consistent ordering
        corridors.sort(key=lambda c: c.idx if c.idx is not None else float('inf'))
        
        print(f"{Fore.GREEN} Successfully loaded {len(corridors)} corridors from folder: {folder_path}{Style.RESET_ALL}")
        return corridors
    
    @staticmethod
    def _is_corridor_file(file_path: str) -> bool:
        """
        Check if a file appears to be a corridor file by looking for expected parameters.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the file appears to be a corridor file
        """
        required_params = {'wp1', 'wp2', 'dir1', 'radius', 'angle', 'width'}
        found_params = set()
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        param_name = line[1:bracket_end]
                        found_params.add(param_name)
                        
                        # If we found all required params, it's a corridor file
                        if required_params.issubset(found_params):
                            return True
        except Exception:
            return False
        
        return False


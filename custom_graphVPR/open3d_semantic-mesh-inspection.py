import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d #version: 0.10.0.0

kimera_ros_path = "../kimera_semantics_ros/"
mesh_prefix = "../kimera_semantics_ros/mesh_results/tesse_shubodh_Inspiron_15_7000_Gaming_" 

def read_cfg_csv(filename):
    #If filename is txt of the form: category, red, green, blue, alpha, id
    # Output will be a dict of the form:
    #category: [red, green, blue, alpha, id]

    dict1 = {}
    no_lines = 0
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if line:
                line_parts = line.split(',')
                cat, red, green, blue  = line_parts[0], line_parts[1], line_parts[2], line_parts[3]
                alpha, id = line_parts[4], line_parts[5]

                dict1[cat] = [red, green, blue, alpha, id]
                no_lines += 1

    #remove 1st key (fields) to avoid confusion
    dict1.pop('name')
    return dict1, no_lines

def fulldict_to_mapping(full_dict):
    '''
    Input: full_dict i.e. full text file in the form of dict.     
    Output: dict(key, value) where key is unique [rgb] number for every category (value).

    The task here is to extract semantic_category_name (like Books) from the semantic mesh.
    The mesh has unique rgb values corresponding to which there exists a unique semantic_category_name. 

    So the simple idea is:
    r + g + b must be unique, so let's first create a dict of key as (r+g+b) 
    and value as semantic_category_name from full_dict aka full text file. This is what 
    we're doing in this function.

    Then, given the semantic mesh's rgb, we can easily extract its semantic_category_name,
    doing ths in function `extract_mesh_labels()`.
    '''
    dict_map = {}

    for key, val in full_dict.items():
        r_, g_, b_, alp, id = val 
        rgb = r_ + g_ + b_
        # Don't get confused: key of input is value of output
        dict_map.setdefault(rgb, [])
        dict_map[rgb].append(key)

        #dict_map[rgb] = key 
    
    # Added the following manually because of strange behaviour in mesh file:
    # It says r,g,b of 255,255,255 is there in mesh, but this isn't available in our full txt file.
    # I don't understand: Why would you label your colour as 255,255,255... Not sure, but added so that code runs.
    dict_map['255255255'] = 'dummy'

    print(f"\nWarning: Added `dummy` category manually. Might face issues later, keep this in mind for future tasks. \n")
    
    return dict_map

def extract_mesh_labels(semantic_pcd_filename, rgb_to_cat):
    pcd = o3d.io.read_point_cloud(semantic_pcd_filename)
    pcd_colors = np.asarray(pcd.colors)
    uniq_clr, indices = np.unique(pcd_colors*255, axis=0, return_index=True)
    labels = []
    dict_labels = {}
    for i in range(uniq_clr.shape[0]):
        query_rgb = str(int(uniq_clr[i,0]))+ str(int(uniq_clr[i,1]))+ str(int(uniq_clr[i,2]))
        vals = rgb_to_cat[query_rgb]

        dict_labels.setdefault(query_rgb, [])
        dict_labels[query_rgb].append(vals)

        labels.append(vals)

    num_of_instances = 0
    for list_i in labels:
        num_of_instances += len(list_i)
    return labels, dict_labels, num_of_instances

def pcd_info(filename, viz=False):
    pcd = o3d.io.read_point_cloud(filename)
    if viz == True:
        o3d.visualization.draw_geometries([pcd])
    pcd_colors = np.asarray(pcd.colors)
    uniq_clr, indices = np.unique(pcd_colors*255, axis=0, return_index=True)
    
    return uniq_clr.shape[0] # Number of unique semantic categories in mesh


def pcd_show_cat(filename, cat, viz=False):
    '''
    Input: Filename along with category (r+g+b)
    Output: Segmented out pcd + visualization of output if viz=True
    '''
    pcd = o3d.io.read_point_cloud(filename)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    pcd_colors = pcd_colors*255
    pcd_colors_ids = np.zeros(pcd_colors.shape[0])
    for i in range(pcd_colors.shape[0]):
        pcd_colors_ids[i] = str(int(pcd_colors[i,0]))+ str(int(pcd_colors[i,1]))+ str(int(pcd_colors[i,2]))
    pcd_colors_ids_cat = np.argwhere(pcd_colors_ids==float(cat))
    pcd_points_cat = np.squeeze(pcd_points[pcd_colors_ids_cat], axis=1)
    pcd_colors_cat = np.squeeze(pcd_colors[pcd_colors_ids_cat], axis=1)
    pcd_output = o3d.geometry.PointCloud()
    pcd_output.points = o3d.utility.Vector3dVector(pcd_points_cat)
    #pcd_output.colors = o3d.utility.Vector3dVector(pcd_colors_cat) #TODO: Doesn't seem to work in all cases.. Want to show it as per original colors
    
    if viz == True:
        print(pcd_points_cat.shape)
        print(f"\n\nSHOWING FULL PCD:\n\n")
        o3d.visualization.draw_geometries([pcd])
        print(f"\n\nSHOWING SEGMENTED PCD with R+G+B ID - {str(cat)}:\n\n")
        o3d.visualization.draw_geometries([pcd_output])
    return pcd_output 

def dbscan_clustering(pcd):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.28, min_points=50, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def print_general(num_sem_cat_mesh, rgb_to_cat, num_instances, num_lines):
    print(f"\n\nPRINTING OUT GENERAL DETAILS:\n\n")
    print(f"\nNumber of unique semantic categories in mesh, i.e. ESTIMATED, are {num_sem_cat_mesh}. \n")
    print(f"Number of unique semantic categories in given csv file, i.e. GROUND TRUTH, are {len(rgb_to_cat)}. \n")
    print(f"\nNumber of instance categories in mesh, i.e. ESTIMATED instances*, are {num_instances}. \n")
    print(
    '''
    * Do note that without applying any clustering, there is no way to find num of estimated instances.
    The above 'ESTIMATED instances' just includes all categories with a unique R+G+B value from the txt file.
    That does not mean all those instances were actually found. So it can be thought of as an upper bound value.
    ''')
    print(f"\nNumber of instance categories in given csv file, i.e. TOTAL instances**, are {num_lines + 1}. \n")
    print(f"     ** added 1 for manually added dummy category. See fulldict_to_mapping() function.\n")

if __name__ == '__main__':
    mesh_name = "26094_3367494325221832115.ply"

    num_sem_cat_mesh = pcd_info(mesh_prefix + mesh_name, viz=False)
    full_dict, num_lines = read_cfg_csv(kimera_ros_path + "cfg/tesse_multiscene_office1_segmentation_mapping.csv") 
    rgb_to_cat = fulldict_to_mapping(full_dict)
    _, dict_meshlabels, num_instances = extract_mesh_labels(mesh_prefix + mesh_name, rgb_to_cat)

    #print_general(num_sem_cat_mesh, rgb_to_cat, num_instances, num_lines)

    Chairs = 1022550; Floor = 124133141; Table = 54176239
    pcd_segmented = pcd_show_cat(mesh_prefix+mesh_name, Chairs, viz=False)

    dbscan_clustering(pcd_segmented)
    #TODO: Just using Open3D's dbscan for clustering. However, the remaining task is to 
    # do what is suggested exactly in the 3DSceneGraphs paper (using PCL library, not Open3D)
    # and accurately go from semantic segmentation to instance segmentation.
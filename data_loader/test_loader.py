from torch.utils.data import Dataset
import numpy as np
import trimesh
import torch
import os
from scipy import spatial
from glob import glob 
from utils.utils import *

            


def get_corresponding_points(target,normals, gar_emd, target_emd, device = None, knn=3, delta = 0.02):
    embedsize = 128 
    smpl_tree = spatial.cKDTree(target_emd.cpu()[:,:embedsize])
    nn_dist, nn_idx = smpl_tree.query(gar_emd.cpu()[:,:embedsize], k=knn)

    nn_dist = nn_dist.reshape(-1,knn)
    nn_idx = nn_idx.reshape(-1,knn)
        
    dist = nn_dist                   # N, KNN_Neighbors
    weights = (1/dist)[..., None]    # N, KNN_Neightbors, 1   
    weights_sum = np.sum(weights,1)  # N, 1
    weighted = torch.sum(target[nn_idx] * weights, 1) # target = (N , KNN_N, 3), weights = (N, KNN_N, 1) => weighted = (N,3)
    pose3d = weighted/weights_sum                   # pose3d N 3

    normals_w = torch.sum(normals[nn_idx] * weights, 1) # target = (N , KNN_N, 3), weights = (N, KNN_N, 1) => weighted = (N,3)
    normals_w = normals_w/weights_sum                  # normal N 3
    l2_norm = torch.linalg.norm(normals_w[..., None], 2, 1) # ord=2, axis=1, l2_norm => N, 1
    l2_norm[l2_norm<1e-5] = 1e-5
    normals_normalized = normals_w / l2_norm 

    weighted_mask = (weights_sum-weights_sum.min(0))/(weights_sum.max(0)-weights_sum.min(0))
    pose3d = pose3d + normals_normalized*delta
    return pose3d, normals_normalized, weighted_mask, nn_dist



def get_smpl_neighboring_points(gar,target, gar_emd, target_emd, device = None, knn=32):
    smpl_tree = spatial.cKDTree(target_emd.cpu())
    nn_dist, nn_idx = smpl_tree.query(gar_emd.cpu(), k=knn)
    
    weighted = target[nn_idx]/np.expand_dims(nn_dist,-1)
    dist_sum = np.sum(1/nn_dist,-1)
    normalized = weighted/dist_sum.reshape(dist_sum.shape[0],1,1)
    normalized = normalized.reshape([-1,knn*3])
    pose3d = torch.cat([gar,normalized],axis = -1)
    return pose3d

   
   
class TestDataset(Dataset):
    def __init__(self, opt = None, knn = 32):
        self.knn = knn
        self.device = opt.device
        self.opt = opt
        self.smpl_path = opt.smpl_path
        self.smpl_emd_path = opt.smpl_emd
        
        
        #load target data
        self.smpl_meshes = []
        self.smpl_names = []
        self.smpl_vertices = []
        self.smpl_faces = []
        self.smpl_normals = []
        self.smpl_shapes = []
        self.smpl_poses = []
        self.smpl_corresp_pos = []
        self.smpl_corresp_normals = []
        self.smpl_corresp_mask = []
        self.smpl_corresp_dist = []
        self.model_input = []
        self.garment_idx = []
        
        #garment data
        self.gar_verts = []
        self.gar_face_verts = []
        self.gar_faces = []
        self.gar_adj_faces = []
        self.gar_adj_fedges = []
        self.gar_emds = []
        self.gar_names = []
        self.gar_edges = []
        self.gar_color = []
        
        # load garment data
        if(opt.garment_name):
            gar_mesh_paths = [os.path.join(opt.garment_path,'mesh',f'{opt.garment_name}.obj') ]   
        else:
            gar_mesh_paths = glob(os.path.join(opt.garment_path,'mesh','*'))

        for gar_idx,garment_path in enumerate(gar_mesh_paths):  
            #garment mesh
            garment_name = garment_path.split('/')[-1].split('.')[0] 
            garment_emd_path = f'{opt.garment_path}/embedding/{garment_name}.npz'
            

            garment_mesh = trimesh.load(garment_path,process=False,maintain_order=True)  
            garment_vertices = np.asarray(garment_mesh.vertices) 
            
            
            self.gar_names.append(garment_name)   
            self.gar_verts.append(garment_vertices)  
            garment_faces = np.asarray(garment_mesh.faces).astype(np.int64)
            garment_edges = np.asarray(garment_mesh.edges).astype(np.int64)
            garment_adj_faces =  np.asarray(garment_mesh.face_adjacency).astype(np.int64)
            garment_adj_fedges =  np.asarray(garment_mesh.face_adjacency_edges).astype(np.int64)

            
                    
            #embedding
            garment_emd = np.load(garment_emd_path)
            if(garment_emd_path.split('/')[-1].split('.')[-1]=='npz'):
                garment_emd = garment_emd['arr_0']   #for THumans,3dhumans, comment for synthetic
                
            garment_emd = torch.Tensor(garment_emd).type(torch.float32).to(self.device)
            self.gar_emds.append(garment_emd)
        
            
            face_vertices = []
            for j in range(garment_faces.shape[0]):
                face_vertices.append([garment_vertices[garment_faces[j][0]], garment_vertices[garment_faces[j][1]], garment_vertices[garment_faces[j][2]]])
            
            garment_vertices = torch.Tensor(garment_vertices).type(torch.float32)
            
            garment_faces = torch.Tensor(garment_faces).type(torch.int64)
            face_vertices = torch.Tensor(np.array(face_vertices,dtype=np.float32))
            self.gar_faces.append(garment_faces)
            self.gar_edges.append(garment_edges)
            self.gar_face_verts.append(face_vertices)
            self.gar_adj_faces.append(garment_adj_faces)
            self.gar_adj_fedges.append(garment_adj_fedges)

        
        
            #taget_smpl_meshes
            if(opt.real_scan):
                smpl_mesh_paths = glob(os.path.join(self.smpl_path, 'mesh','real_scans', '*'))
            else:
                smpl_mesh_paths = glob(os.path.join(self.smpl_path, 'mesh','smpl_data', '*'))

            for mesh in smpl_mesh_paths: 
                self.garment_idx.append(gar_idx)
                target_path = mesh   #3dhumans,amass

                if(opt.real_scan):
                    target_mesh = trimesh.load(target_path,process=False, maintain_order=True)
                    target_faces = torch.Tensor(target_mesh.faces).type(torch.long)
                    
                    
                else:  
                    target_mesh = trimesh.load(target_path, maintain_order=True, process=False)
                    target_faces = np.array(target_mesh.faces).astype(np.int64)
                

                name = mesh.split('/')[-1].split('.')[0]
                #normalize
                verts = np.asarray(target_mesh.vertices)
                
            
            
                target_verts = np.array(verts)
                target_verts = torch.Tensor(target_verts).type(torch.float32)
                target_normals = np.array(target_mesh.vertex_normals)      
                target_normals = torch.Tensor(target_normals).type(torch.float32)      
                
                #smpl_emd 
                if(opt.real_scan):
                    self.smpl_emd = np.load(os.path.join(opt.smpl_path,'embedding',f'{name}.npz')  )['arr_0']
                else:
                    self.smpl_emd = np.load(self.smpl_emd_path)
                    
                self.smpl_emd = torch.Tensor(self.smpl_emd).type(torch.float64).to(self.device)
                

                
                
                #append to list
                self.smpl_meshes.append(target_mesh)
                self.smpl_names.append(name)
                self.smpl_vertices.append(target_verts)
                self.smpl_faces.append(target_faces)
                self.smpl_normals.append(target_normals)
                
                
                if(self.opt.real_scan):
                    delta = 0.03
                else:
                    delta = 0.02
                #correspondence 
                corresp_points,corresp_normals,corresp_mask,corresp_dist = get_corresponding_points(target_verts, target_normals, garment_emd, self.smpl_emd, delta=delta)
                garment_mesh.vertices = corresp_points
                try:
                    garment_mesh = laplacian_smoothning(garment_mesh, 5)
                except: 
                    print("Embedding not working")
                
                
                corresp_points = garment_mesh.vertices
                
                corresp_points = torch.Tensor(corresp_points).type(torch.float32)
                corresp_normals = torch.Tensor(corresp_normals).type(torch.float32)
                corresp_dist = torch.Tensor(corresp_dist).type(torch.float32)
                
                #append smpl verts to garments
                gar_corresp_points = get_smpl_neighboring_points(corresp_points, target_verts, garment_emd, self.smpl_emd)            
                gar_corresp_points = torch.Tensor(gar_corresp_points).type(torch.float32).to(self.device)

                #concatenate iso emd
                gar_corresp_points = torch.cat([gar_corresp_points, garment_emd],axis=1)
                self.model_input.append(gar_corresp_points)
                
                #append smpl data
                self.smpl_corresp_pos.append(corresp_points)
                self.smpl_corresp_normals.append(corresp_normals)
                self.smpl_corresp_mask.append(corresp_mask)
                self.smpl_corresp_dist.append(corresp_dist)
    
        
         

    def __getitem__(self, idx):
        body = {}
        body['gt'] = self.smpl_corresp_pos[idx]
        body['gt_normals'] = self.smpl_corresp_normals[idx]
        body['model_input'] = self.model_input[idx]
        body['target_verts'] = self.smpl_vertices[idx]
        body['target_faces'] = self.smpl_faces[idx]
        body['target_normals'] = self.smpl_normals[idx]
        body['target_embedding'] = self.smpl_emd
        body['target_name'] = self.smpl_names[idx]
        gar_idx = self.garment_idx[idx]
        
        body['gar_verts'] = torch.Tensor(self.gar_verts[gar_idx]).type(torch.float32)
        body['gar_faces'] = self.gar_faces[gar_idx]
        body['gar_embedding'] = self.gar_emds[gar_idx]
        body['gar_name'] = self.gar_names[gar_idx]
        body['gar_edges'] = self.gar_edges[gar_idx]

        return body
        

    def __len__(self):
        return len(self.smpl_meshes)


if __name__ == '__main__':
    dataset = TestDataset()
    print(dir(dataset[0]))
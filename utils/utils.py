import numpy as np

def laplacian_smoothning(mesh, iterations=1):
    verts = mesh.vertices
    ring_n = mesh.vertex_neighbors
    n_verts = verts.shape[0] 
    mask = np.zeros((n_verts,n_verts))
    for nidx in range(n_verts):
        mask[nidx, ring_n[nidx]] = 1
        if(len(ring_n[nidx])>0):
            mask[nidx, ring_n[nidx]] /= len(ring_n[nidx])
                
    for i in range(iterations):
        verts = np.matmul(mask,verts) 
    
    mesh.vertices = verts
    return mesh
                 
def load_obj(fname, quad=False):
    
    f = open(fname)

    v = []
    vt = []
    vn = []
    f_v = []
    f_vt = []
    f_vn = []

    for line in f:

        if line[0:2]=='v ':
            info = line.strip().split(" ")
            v.append([float(info[1]),float(info[2]),float(info[3])])
        
        elif line[0:2]=='vt':
            info = line.strip().split(" ")
            vt.append([float(info[1]),float(info[2])])
            
        elif line[0:2]=='vn':
            info = line.strip().split(" ")
            vn.append([float(info[1]),float(info[2]),float(info[3])])

        elif line[0:2]=='f ':

            info = line.strip().split(" ")
            face1 = info[1].split("/")
            face2 = info[2].split("/")
            face3 = info[3].split("/")

            face4=None
            if quad:
                face4 = info[4].split("/")

            if quad:
                verts_idx = [int(face1[0]),int(face2[0]),int(face3[0]),int(face4[0])]
            else:
                verts_idx = [int(face1[0]),int(face2[0]),int(face3[0])]
            f_v.append(verts_idx)

            if len(vt)>0:
                if quad:
                    verts_tex_idx = [int(face1[1]),int(face2[1]),int(face3[1]),int(face4[1])]
                else:
                    verts_tex_idx = [int(face1[1]),int(face2[1]),int(face3[1])]
                f_vt.append(verts_tex_idx)

            if len(vn)>0:
                if quad:
                    verts_norm_idx = [int(face1[2]),int(face2[2]),int(face3[2]),int(face4[2])]
                else:
                    verts_norm_idx = [int(face1[2]),int(face2[2]),int(face3[2])]
                f_vn.append(verts_norm_idx)


    v = np.array(v)
    vt = np.array(vt)
    vn = np.array(vn)
    f_v = np.array(f_v)
    f_vt = np.array(f_vt)
    f_vn = np.array(f_vn)

    mesh_info = {'v':v,'vt':vt,'vn':vn,'f_v':f_v,'f_vt':f_vt,'f_vn':f_vn}

    return mesh_info



def write_obj(original_mesh, original_tri, reposed_mesh, final_mesh_path, quad=False):
    f = open(final_mesh_path, 'w') # creatng textured mesh
    verts = reposed_mesh['v']
    verts_tex = original_mesh['vt']
    # verts_tex = (verts_tex - verts_texcd /da  .min(0)) / (verts_tex.max(0) - verts_tex.min(0))
    f_v = original_tri['f_v']
    f_vt = original_mesh['f_vt']
    for v in verts:
        f.write('v '+str(v[0])+' '+str(v[1])+' '+str(v[2])+'\n')
    for vt in verts_tex:
        f.write('vt '+str(vt[0])+' '+str(vt[1])+'\n')
    for fidx in range(len(f_v)):
        v_f = f_v[fidx]
        vt_f = f_vt[fidx]
        if quad:
            f.write('f '+str(v_f[0])+'/'+str(vt_f[0])+' '+str(v_f[1])+'/'+str(vt_f[1])+' '+str(v_f[2])+'/'+str(vt_f[2]) +' '+str(v_f[3])+'/'+str(vt_f[3]) + '\n')
        else:
            f.write('f '+str(v_f[0])+'/'+str(vt_f[0])+' '+str(v_f[1])+'/'+str(vt_f[1])+' '+str(v_f[2])+'/'+str(vt_f[2]) + '\n')
        
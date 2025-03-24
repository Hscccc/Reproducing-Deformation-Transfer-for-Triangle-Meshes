import openmesh as om
import numpy as np
import scipy.linalg

def cal_Vi(a, b, c):
    d = np.cross(b - a, c - a)
    d = d / np.linalg.norm(d)
    V = np.array([b - a, c - a, d])
    return V.T

def get_face_vertex_indices(mesh, face_idx):
    f = mesh.face_handle(face_idx)
    return [vh.idx() for vh in mesh.fv(f)]

if __name__ == "__main__":
    # read files
    s0 = om.TriMesh()
    s1 = om.TriMesh()
    t0 = om.TriMesh()
    t1 = om.TriMesh()

    s1 = om.read_trimesh("obj/s1.obj")
    s0 = om.read_trimesh("obj/s0.obj")
    t0 = om.read_trimesh("obj/t0.obj")

    n = s0.n_vertices()
    m = s0.n_faces()

    # calculate b
    b = np.zeros((3 * m, 3))
    for i in range(m):
        f = s0.face_handle(i)
        v = [s0.point(vh) for vh in s0.fv(f)]
        S = cal_Vi(v[0], v[1], v[2])

        f = s1.face_handle(i)
        v = [s1.point(vh) for vh in s1.fv(f)]
        S_ = cal_Vi(v[0], v[1], v[2])

        C = np.dot(S_, np.linalg.inv(S))
        b[3 * i: 3 * i + 3] = C.T
    
    # calculate A
    A = np.zeros((3 * m, m + n))
    for i in range(m):
        f = t0.face_handle(i)
        v = [t0.point(vh) for vh in t0.fv(f)]
        S = cal_Vi(v[0], v[1], v[2])

        v_index = get_face_vertex_indices(t0, i)
        W = np.zeros((n + m, 3))
        W[v_index[0]] = np.array([-1, -1, -1])
        W[v_index[1]] = np.array([1, 0, 0])
        W[v_index[2]] = np.array([0, 1, 0])
        W[n + i] = np.array([0, 0, 1])
        A[3 * i:3 * i + 3] = np.dot(W, np.linalg.inv(S)).T


    # solve using LU decomposition
    QQQ = A.T @ A
    WWW = A.T @ b
    P, L, U = scipy.linalg.lu(QQQ)
    y = np.linalg.solve(L, P.T @ WWW)
    x = np.linalg.solve(U, y)

    # output t1.obj
    for i in range(n):
        t1.add_vertex(x[i])
    for i in range(m):
        f = t0.face_handle(i)
        v_index = get_face_vertex_indices(t0, i)
        v = [t1.vertex_handle(v_index[j]) for j in range(3)]
        t1.add_face(v[0], v[1], v[2])
        
    om.write_mesh("obj/t1.obj", t1)

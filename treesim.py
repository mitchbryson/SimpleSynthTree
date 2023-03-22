"""
treesim.py: Functions for tree model simulation and sampling
Author: Mitch Bryson

Requires NumPy and Trimesh
"""

import os, sys
from math import *
import trimesh
import numpy as np

# get_spline_params: get spline parameters from points
def get_spline_params(p0, p1, p2):
    A = np.array([ [2/(p1[0]-p0[0]), 1/(p1[0]-p0[0]), 0], \
        [1/(p1[0]-p0[0]), 2*((1/(p1[0]-p0[0]))+(1/(p2[0]-p1[0]))), 1/(p2[0]-p1[0])], \
        [0, 1/(p2[0]-p1[0]), 2/(p2[0]-p1[0])] ])
    b = np.array([3*(p1[1]-p0[1])/pow(p1[0]-p0[0],2), \
        3*( (p1[1]-p0[1])/pow(p1[0]-p0[0],2) + (p2[1]-p1[1])/pow(p2[0]-p1[0],2) ), \
        3*(p2[1]-p1[1])/pow(p2[0]-p1[0],2)])
    k = np.linalg.solve(A, b)
    return k

# get_spline_points: get points along a spline defined by points and parameters kx, ky
def get_spline_points(p_spline,kx,ky,z):
	
	x0 = p_spline[0][0]
	y0 = p_spline[0][1]
	z0 = p_spline[0][2]
	x1 = p_spline[1][0]
	y1 = p_spline[1][1]
	z1 = p_spline[1][2]
	x2 = p_spline[2][0]
	y2 = p_spline[2][1]
	z2 = p_spline[2][2]
	
	a01x = kx[0]*(z1-z0) - (x1-x0);
	a12x = kx[1]*(z2-z1) - (x2-x1);
	a01y = ky[0]*(z1-z0) - (y1-y0);
	a12y = ky[1]*(z2-z1) - (y2-y1);
	
	b01x = -kx[1]*(z1-z0) - (x1-x0);
	b12x = -kx[2]*(z2-z1) - (x2-x1);
	b01y = -ky[1]*(z1-z0) - (y1-y0);
	b12y = -ky[2]*(z2-z1) - (y2-y1);
	
	t01 = (z-z0)/(z1-z0)
	t12 = (z-z1)/(z2-z1)
	
	x01out = x0*(1-t01) + x1*t01 + np.multiply(np.multiply(t01,(1-t01)),(a01x*(1-t01) + b01x*t01))
	x12out = x1*(1-t12) + x2*t12 + np.multiply(np.multiply(t12,(1-t12)),(a12x*(1-t12) + b12x*t12))
	y01out = y0*(1-t01) + y1*t01 + np.multiply(np.multiply(t01,(1-t01)),(a01y*(1-t01) + b01y*t01))
	y12out = y1*(1-t12) + y2*t12 + np.multiply(np.multiply(t12,(1-t12)),(a12y*(1-t12) + b12y*t12))
	
	xout = np.multiply(x01out,(z < z1)) + np.multiply(x12out,(z >= z1))
	yout = np.multiply(y01out,(z < z1)) + np.multiply(y12out,(z >= z1))
	
	return (xout,yout)

# gen_single_simtree: generate synthetic trees
def gen_single_simtree(Nstem=300, Nfol=1000, model_params=None, tx=None, ty=None):
    
    # class IDs used for different point types
    class_id = {}
    class_id['folliage'] = 0
    class_id['stem'] = 1
    
    # Generate main stem parameters
    if tx is None:
        tx = 0
    if ty is None:
        ty = 0
    tz = 0
    hr = model_params['height_range']
    h = hr[0] + (hr[1]-hr[0])*np.random.random()
    dr = model_params['diam_range']
    r = 0.5*dr[0] + 0.5*(dr[1]-dr[0])*np.random.random()
    
    # randomise fork status
    rn = np.random.random()
    if rn < model_params['split_prob']:
        fork = 2
    else:
        fork = 1
    
    stem_data = []
    
    # Initial spline for main stem
    p0 = [tx,ty,tz]
    sweeptopx = 2*model_params['tree_top_dist']*np.random.random()-model_params['tree_top_dist']
    sweeptopy = 2*model_params['tree_top_dist']*np.random.random()-model_params['tree_top_dist']
    tmdist = model_params['tree_mid_dist']
    p1 = [tx+2*tmdist*np.random.random()-tmdist+0.2*sweeptopx,ty+2*tmdist*np.random.random()-tmdist+0.2*sweeptopy,0.5*h+(0.167*h*np.random.random())]
    p2 = [tx+sweeptopx,ty+sweeptopy,h]
    p_spline = np.array([p0, p1, p2])
    kx = get_spline_params( (p_spline[0][2],p_spline[0][0]), (p_spline[1][2],p_spline[1][0]), (p_spline[2][2],p_spline[2][0]))
    ky = get_spline_params( (p_spline[0][2],p_spline[0][1]), (p_spline[1][2],p_spline[1][1]), (p_spline[2][2],p_spline[2][1]))
    
    if fork == 1: # Single stem
        
        zc = np.linspace(tz,h,num=20)
        (xc,yc) = get_spline_points(p_spline,kx,ky,zc)
        stem_data.append([xc,yc,zc,p_spline,kx,ky,[0,h]])
    
    elif fork == 2: # Split/forked stem
        
        # Generate parameters for forks
        splhr = model_params['split_height_range']
        hf = h*(splhr[0]+(splhr[1]-splhr[0])*np.random.random())
        
        zc = np.linspace(tz,hf,num=20)
        (xc,yc) = get_spline_points(p_spline,kx,ky,zc)
        stem_data.append([xc,yc,zc,p_spline,kx,ky,[0,hf]])
        
        dr = 2.0*np.random.random()+1.5
        theta = 2*pi*np.random.random()
        dx = dr*sin(theta)
        dy = dr*cos(theta)
        topxy1 = [sweeptopx+dx,sweeptopy+dy,h]
        l2 = (0.3*np.random.random()+0.7)*(h-hf)+hf
        
        topxy2 = [sweeptopx-dx,sweeptopy-dy,l2]
        
        bottomxy = [xc[-1],yc[-1],hf]
        dx1 = 0.3*(topxy1[0]-bottomxy[0])
        dy1 = 0.3*(topxy1[1]-bottomxy[1])
        dx2 = 0.3*(topxy2[0]-bottomxy[0])
        dy2 = 0.3*(topxy2[1]-bottomxy[1])
        bend = 0.3+0.1*np.random.random()
        midxy1 = [bottomxy[0]+dx1,bottomxy[1]+dy1,bend*(h-hf)+hf]
        midxy2 = [bottomxy[0]+dx2,bottomxy[1]+dy2,bend*(topxy2[2]-hf)+hf]
        
        p_spline = np.array([bottomxy, midxy1, topxy1])
        kx = get_spline_params( (p_spline[0][2],p_spline[0][0]), (p_spline[1][2],p_spline[1][0]), (p_spline[2][2],p_spline[2][0]))
        ky = get_spline_params( (p_spline[0][2],p_spline[0][1]), (p_spline[1][2],p_spline[1][1]), (p_spline[2][2],p_spline[2][1]))
        zc = np.linspace(hf,topxy1[2],num=20)
        (xc,yc) = get_spline_points(p_spline,kx,ky,zc)
        stem_data.append([xc,yc,zc,p_spline,kx,ky,[hf,topxy1[2]]])
        
        p_spline = np.array([bottomxy, midxy2, topxy2])
        kx = get_spline_params( (p_spline[0][2],p_spline[0][0]), (p_spline[1][2],p_spline[1][0]), (p_spline[2][2],p_spline[2][0]))
        ky = get_spline_params( (p_spline[0][2],p_spline[0][1]), (p_spline[1][2],p_spline[1][1]), (p_spline[2][2],p_spline[2][1]))
        zc = np.linspace(hf,topxy2[2],num=20)
        (xc,yc) = get_spline_points(p_spline,kx,ky,zc)
        stem_data.append([xc,yc,zc,p_spline,kx,ky,[hf,topxy2[2]]])
        
    # Generate stem mesh
    verts = []
    faces = []
    lv = 0
    for stem in stem_data:
        (xc,yc,zc,p,kx,ky,zrange) = stem
        Ndotsl = 12
        t = np.linspace(0,2*pi,Ndotsl+1)
        t = t[:-1]
        for i in range(len(zc)):
            rn = r*(1-(zc[i]/h))
            xn = rn*np.sin(t)+xc[i]
            yn = rn*np.cos(t)+yc[i]
            for j in range(xn.shape[0]):
                verts.append([xn[j],yn[j],zc[i]])
        for i in range(len(zc)-1):
            for j in range(t.shape[0]-1):
                faces.append([Ndotsl*i+j+1+lv,Ndotsl*i+j+lv,Ndotsl*(i+1)+j+lv,Ndotsl*(i+1)+j+1+lv])
            faces.append([Ndotsl*i+lv,Ndotsl*i+t.shape[0]-1+lv,Ndotsl*(i+1)+t.shape[0]-1+lv,Ndotsl*(i+1)+lv])
        lv = len(verts)
    
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    
    # Sample points from stem meshes
    points = mesh.sample(Nstem)
    points_all = np.concatenate((points,class_id['stem']*np.ones((Nstem,1))),axis=1)
    
    # Generate branches and foliage
    h1r = [model_params['min_can_height'][0], model_params['min_can_height'][1]]
    h2r = [model_params['max_can_width_height'][0], model_params['max_can_width_height'][1]]
    h1 = h*(h1r[0]+(h1r[1]-h1r[0])*np.random.random())
    h2 = h*(h2r[0]+(h2r[1]-h2r[0])*np.random.random())
    
    branchz1 = h1+(h2-h1)*np.random.random(np.random.randint(0.166*model_params['num_branches'][0],0.3*model_params['num_branches'][1]))
    branchz2 = h2+(h-h2)*np.random.random(np.random.randint(fork*0.833*model_params['num_branches'][0],fork*0.7*model_params['num_branches'][1]))
    
    branchz = np.concatenate((branchz1,branchz2))
    xyc = []
    for stem in stem_data:
        (x,y,z,p_spline,kx,ky,zrange) = stem
        (xc,yc) = get_spline_points(p_spline,kx,ky,branchz)
        xyc.append((xc,yc))
    
    # Generate mesh data for branches and foliage
    verts = []
    faces = []
    c = 0
    rolloff = 1.0
    rolloffh = h2
    rolloffh2 = h1
    for i in range(len(branchz)):
        
        # determine which stem to put on
        okstems = []
        for (j,stem) in enumerate(stem_data):
            (x,y,z,p_spline,kx,ky,zrange) = stem
            if branchz[i] >= zrange[0] and branchz[i] <= zrange[1]:
                okstems.append(j)
        if len(okstems) == 0:
            pass
        elif len(okstems) == 1:
            j = okstems[0]
            xc = xyc[j][0]
            yc = xyc[j][1]
        else:
            j = okstems[np.random.choice(len(okstems),1)[0]]
            xc = xyc[j][0]
            yc = xyc[j][1]
        
        if (i % 10) == 0:
            gap = np.random.random()
            gap2 = (2*pi-gap)*np.random.random()
        theta = 2*pi*np.random.random()
        if theta > gap2 and theta < (gap2+gap):
            continue
        rn = r*(1-(branchz[i]/h))
        rout = (model_params['max_can_width']/7.0)*sqrt(h-branchz[i])
        if branchz[i] < rolloffh2:
            rout = 0
        elif branchz[i] < rolloffh:
            rout = rout-(rolloff*rout*((rolloffh-branchz[i])/rolloffh))
        
        rout2 = rout+0.4*rout*np.random.random()-0.2*rout
        thick = 0.2*rout2
        verts.append([rn*sin(theta)+xc[i],rn*cos(theta)+yc[i],branchz[i]])
        verts.append([(rout2/3)*sin(theta)+thick*cos(theta)+xc[i],(rout2/3)*cos(theta)-thick*sin(theta)+yc[i],branchz[i]+np.random.random()])
        verts.append([(rout2/3)*sin(theta)-thick*cos(theta)+xc[i],(rout2/3)*cos(theta)+thick*sin(theta)+yc[i],branchz[i]+np.random.random()])
        verts.append([(2*rout2/3)*sin(theta)+thick*cos(theta)+xc[i],(2*rout2/3)*cos(theta)-thick*sin(theta)+yc[i],branchz[i]+np.random.random()])
        verts.append([(2*rout2/3)*sin(theta)-thick*cos(theta)+xc[i],(2*rout2/3)*cos(theta)+thick*sin(theta)+yc[i],branchz[i]+np.random.random()])
        verts.append([rout2*sin(theta)+xc[i],rout2*cos(theta)+yc[i],branchz[i]+np.random.random()])
        
        faces.append([6*c,6*c+1,6*c+2])
        faces.append([6*c+1,6*c+2,6*c+4])
        faces.append([6*c+1,6*c+4,6*c+3])
        faces.append([6*c+3,6*c+4,6*c+5])
        c += 1
    
    mesh2 = trimesh.Trimesh(vertices=verts,faces=faces)
    
    # Sample foliage points
    points = mesh2.sample(Nfol)
    points[:,2] += model_params['foliage_noise']*np.random.random(points.shape[0])
    points_class = np.concatenate((points,class_id['folliage']*np.ones((Nfol,1))),axis=1)
    
    points_all = np.concatenate((points_all,points_class))
    
    return points_all

# gen_simtree: Generates sample point cloud of synthetically-generated tree
def gen_simtree(Np=1024, stem_frac_lambda=0.3, model_params=None):
    
    # create example default parameters if none specified
    if model_params is None:
        model_params = {}
        model_params['height_range'] = [30, 50]
        model_params['diam_range'] = [0.5, 1]
        model_params['split_height_range'] = [0.15, 0.5]
        model_params['split_prob'] = 0.5
        model_params['num_branches'] = [60, 100]
        model_params['min_can_height'] = [0.2, 0.5]
        model_params['max_can_width'] = 7.0
        model_params['max_can_width_height'] = [0.4, 0.8]
        model_params['tree_top_dist'] = 2.5
        model_params['tree_mid_dist'] = 0.5
        model_params['foliage_noise'] = 0.5
    
    # stem_frac_lambda: fraction of stem hits relative to foliage
    if isinstance(stem_frac_lambda,list): # stem_frac within a specified range
        stem_frac_lambda = stem_frac_lambda[0] + np.random.random()*(stem_frac_lambda[1]-stem_frac_lambda[0])
    
    # Work out numbers of points
    Nground = 0
    Nstem = int(stem_frac_lambda*Np)
    Nfol = Np-(Nground+Nstem)
    
    # Generate initial point cloud
    points_all = np.zeros((0,4))
    
    # Generate stem and foliage
    points_tree = gen_single_simtree(Nstem=Nstem,Nfol=Nfol,model_params=model_params)
    
    points_all = np.concatenate((points_all,points_tree))
    
    return points_all



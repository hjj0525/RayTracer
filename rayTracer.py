#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma;
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

class View:
    def __init__(self, viewPoint, viewDir, projNormal, viewUp, projDistance, viewWidth, viewHeight):
        self.viewPoint = viewPoint
        self.viewDir = viewDir
        self.projNormal = projNormal
        self.viewUp = viewUp
        self.projDistance = projDistance
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight

class Shader:
    def __init__(self, shaderType, diffuseColor, specularColor, exponent):
        self.shaderType = shaderType # Lambertian, Phong
        self.diffuseColor = diffuseColor
        self.specularColor = specularColor
        self.exponent = exponent

class Sphere:
    def __init__(self, center, radius, shader):
        self.center = center
        self.radius = radius
        self.shader = shader
    
    def intersect(surface, ray, viewPoint, hitSurface, surfaceIdx, t):
        surfaceIdx += 1 # idx 한 개 진행
        a = np.sum(ray * ray)
        b = np.sum((viewPoint - surface.center) * ray) # 2b = b' (quadratic eq)
        c = np.sum((viewPoint - surface.center)**2) - surface.radius**2

        discriminant = b**2 - a*c # 판별식 b^2-ac
        
        if discriminant >= 0: # 접하거나,두 점
            t0 = (-b + np.sqrt(discriminant)) / a # 해 1 
            t1 = (-b - np.sqrt(discriminant)) / a # 해 2
            tr = t0 if t0 < t1 else t1 # t root 둘 중 작은 해
            if t >= tr and tr >= 0:
                t = tr
                hitSurface = surfaceIdx # hit 했기 때문에 갱신
        
        return hitSurface, t, surfaceIdx
    
class Box:
    def __init__(self, minPt, maxPt, shader, normals):
        self.minPt = minPt
        self.maxPt = maxPt
        self.shader = shader
        self.normals = normals
        
    def intersect(surface, ray, viewPoint, hitSurface, surfaceIdx, t):
            surfaceIdx += 1 # idx 한 개 진행
            intersection = True # intersect 여부
        
            tMin = (surface.minPt[0]-viewPoint[0])/ray[0]
            tMax = (surface.maxPt[0]-viewPoint[0])/ray[0]
    
            # minPt, maxPt는 box를 그릴 뿐, viewDir에 따라 t는 달라지므로
            if tMin > tMax: # swap
                tMin, tMax = tMax, tMin
                
            tyMin = (surface.minPt[1]-viewPoint[1])/ray[1]
            tyMax = (surface.maxPt[1]-viewPoint[1])/ray[1]
            
            if tyMin > tyMax: # swap
                tyMin, tyMax = tyMax, tyMin
            
            if tMin > tyMax or tyMin > tMax: 
                intersection = False
            
            if tyMin > tMin:
                tMin = tyMin
            if tyMax < tMax:
                tMax = tyMax
            
            tzMin = (surface.minPt[2]-viewPoint[2])/ray[2]
            tzMax = (surface.maxPt[2]-viewPoint[2])/ray[2]
            
            if tzMin > tzMax: # swap
                tzMin, tzMax = tzMax, tzMin
            
            if tMin > tzMax or tzMin > tMax:
                intersection = False
            
            if tzMin > tMin:
                tMin = tzMin
            if tzMax < tMax:
                tMax = tzMax
            
            if intersection == True:
                if t >= tMin:
                    t = tMin
                    hitSurface = surfaceIdx
                    
            return hitSurface, t, surfaceIdx

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity
        
# get unit vector
def getUnitV(vec): 
    norm = np.linalg.norm(vec)
    return vec/norm    

# get normal vectors of a box 
def getNormal(minPt, maxPt):
    # E-F-G-H 위 (위에서 봤을 때 반시계)
    # A-B-C-D 아래 
    pA = np.array([minPt[0],minPt[1],minPt[2]])
    pB = np.array([maxPt[0],minPt[1],minPt[2]])
    pD = np.array([minPt[0],minPt[1],maxPt[2]])
    pE = np.array([minPt[0],maxPt[1],minPt[2]])
    
    normals = [] # 각 면의 normal vec
    # 옆면
    normals.append(getUnitV(pA-pD))
    normals.append(getUnitV(pB-pA))
    normals.append(getUnitV(pD-pA))
    normals.append(getUnitV(pA-pB))
    # 위, 아랫면
    normals.append(getUnitV(pE-pA))
    normals.append(getUnitV(pA-pE))
    
    return normals

# get hitSurface, t     
def intersect(surfaces, ray, viewPoint):
    # initial val
    hitSurface = -1
    surfaceIdx = -1
    t = sys.maxsize # no intersect => infinity t
    
    for surface in surfaces:
        # https://pjreddie.com/media/files/Redmon_Thesis.pdf pg.13
        if surface.__class__.__name__ == "Sphere":  
            hitSurface, t, surfaceIdx = surface.intersect(ray, viewPoint, hitSurface, surfaceIdx, t)
                    
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        elif surface.__class__.__name__ == "Box":
            hitSurface, t, surfaceIdx = surface.intersect(ray, viewPoint, hitSurface, surfaceIdx, t)
        
    return hitSurface, t
      
# shading 
def shade(surfaces, view, ray, lights, hitSurface, t):

    if hitSurface == -1: # no intersection
        return np.array([0,0,0])
    
    else:
        # initialzie
        x, y, z = 0, 0, 0
        nVec = np.array([0,0,0])
        
        if surfaces[hitSurface].__class__.__name__ == "Sphere":
            nPoint = view.viewPoint + t*ray # r(t) = p+td, r(t): 접점
            nVec = nPoint - surfaces[hitSurface].center
            nVec = getUnitV(nVec)

        elif surfaces[hitSurface].__class__.__name__ == "Box":
            nPoint = view.viewPoint + t*ray
            # 각 면이 axis-aligned 이므로, nPoint의 한 점은 분명 x,y,z 중 하나 동일
            # 오차 때문에, 최솟값 찾아, normal Vector 지정
            chk1 = abs(nPoint[2] - surfaces[hitSurface].minPt[2])
            chk2 = abs(nPoint[0] - surfaces[hitSurface].maxPt[0])
            chk3 = abs(nPoint[2] - surfaces[hitSurface].maxPt[2])
            chk4 = abs(nPoint[0] - surfaces[hitSurface].minPt[0])
            chk5 = abs(nPoint[1] - surfaces[hitSurface].maxPt[1])
            chk6 = abs(nPoint[1] - surfaces[hitSurface].minPt[1])
            chk = min(chk1, chk2, chk3, chk4, chk5, chk6)
            
            if chk == chk1:
                nVec = surfaces[hitSurface].normals[0]
            elif chk == chk2:
                nVec = surfaces[hitSurface].normals[1]
            elif chk == chk3:
                nVec = surfaces[hitSurface].normals[2]
            elif chk == chk4:
                nVec = surfaces[hitSurface].normals[3]
            elif chk == chk5:
                nVec = surfaces[hitSurface].normals[4]
            elif chk == chk6:
                nVec = surfaces[hitSurface].normals[5]
                
                
        # 빛에 가리는 물체 있나 확인
        for light in lights:
            v = getUnitV(-t*ray) # eye vector (수업 자료, 방향 주의)
            l = -(t*ray + view.viewPoint) + light.position # light vector (수업 자료, 방향 주의)
            l = getUnitV(l)
            lightHitSurface, tLight = intersect(surfaces, -l, light.position)
            
            # 가리는 물체가 없을 경우 shading
            if lightHitSurface == hitSurface: 
                if surfaces[hitSurface].shader.shaderType == 'Lambertian':
                    x = x + surfaces[hitSurface].shader.diffuseColor[0]*light.intensity[0]*max(0, np.dot(l, nVec))
                    y = y + surfaces[hitSurface].shader.diffuseColor[1]*light.intensity[1]*max(0, np.dot(l, nVec))
                    z = z + surfaces[hitSurface].shader.diffuseColor[2]*light.intensity[2]*max(0, np.dot(l, nVec))
                elif surfaces[hitSurface].shader.shaderType == 'Phong':
                    h = getUnitV(v+l)
                    x = x + surfaces[hitSurface].shader.diffuseColor[0]*light.intensity[0]*max(0, np.dot(l, nVec)) + surfaces[hitSurface].shader.specularColor[0]*light.intensity[0] * max(0, np.dot(nVec, h))**surfaces[hitSurface].shader.exponent[0]
                    y = y + surfaces[hitSurface].shader.diffuseColor[1]*light.intensity[1]*max(0, np.dot(l, nVec)) + surfaces[hitSurface].shader.specularColor[1]*light.intensity[1] * max(0, np.dot(nVec, h))**surfaces[hitSurface].shader.exponent[0]
                    z = z + surfaces[hitSurface].shader.diffuseColor[2]*light.intensity[2]*max(0, np.dot(l, nVec)) + surfaces[hitSurface].shader.specularColor[2]*light.intensity[2] * max(0, np.dot(nVec, h))**surfaces[hitSurface].shader.exponent[0]
             
        res = Color(x,y,z)
        res.gammaCorrect(1.8) # gamma correction, 1.6, 1.8, 2.2 중에 선택
        res = res.toUINT8()
        return res
            
def main():

    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # set default values
    viewDir=np.array([0,0,-1]).astype(np.float)
    viewUp=np.array([0,1,0]).astype(np.float)
    viewProjNormal=-1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth=1.0
    viewHeight=1.0
    projDistance=1.0
    intensity=np.array([1,1,1]).astype(np.float)  # how bright the light is.

    # default values for shading
    diffuseColor = np.array([0,0,0]).astype(np.float)
    specularColor = np.array([0,0,0]).astype(np.float)
    exponent = 0
    
    # array for multi-object
    surfaces = []
    lights = []
    
    imgSize=np.array(root.findtext('image').split()).astype(np.int)
    
    # camera: xml -> class
    for c in root.findall('camera'):
        viewPoint=np.array(c.findtext('viewPoint').split()).astype(np.float)
        viewDir = np.array(c.findtext('viewDir').split()).astype(np.float)
        projNormal = np.array(c.findtext('projNormal').split()).astype(np.float)
        viewUp = np.array(c.findtext('viewUp').split()).astype(np.float)
        if(c.findtext('projDistance')):
            projDistance = np.array(c.findtext('projDistance').split()).astype(np.float)
        viewWidth = np.array(c.findtext('viewWidth').split()).astype(np.float)
        viewHeight = np.array(c.findtext('viewHeight').split()).astype(np.float)
    
    view = View(viewPoint, viewDir, projNormal, viewUp, projDistance, viewWidth, viewHeight)
    
    # surface: xml -> class
    for c in root.findall('surface'):
        type_c = c.get('type')
        # Sphere ---------------------------------------------------------------------------------------------------------------------
        if(type_c == 'Sphere'):
            center = np.array(c.findtext('center').split()).astype(np.float)
            radius = np.array(c.findtext('radius').split()).astype(np.float)
            ref = '' # 내부 변수를 사용하기 위해서 선언
            for child in c:
                if(child.tag == 'shader'): # 태그 shader, center, radius중에서 shader
                    ref = child.get('ref') # shader와 비교할 sphere의 ref 값
            
            for sh in root.findall('shader'):
                if(sh.get('name') == ref): # 현재 sphere의 shader ref와 동일한 것을 바로 찾음
                    shaderType = sh.get('type')
                    diffuseColor = np.array(sh.findtext('diffuseColor').split()).astype(np.float) 
                    if(sh.findtext('specularColor')):
                        specularColor = np.array(sh.findtext('specularColor').split()).astype(np.float)
                    if(sh.findtext('exponent')):
                        exponent = np.array(sh.findtext('exponent').split()).astype(np.float)                    
                    shader = Shader(shaderType, diffuseColor, specularColor, exponent) #shader 저장
                    surfaces.append(Sphere(center, radius, shader)) # shader를 따로 저장하지 않고, surface에 편입하여 shader name 필요없음
        # Box ------------------------------------------------------------------------------------------------------------------------
        elif(type_c == 'Box'):
            minPt = np.array(c.findtext('minPt').split()).astype(np.float)
            maxPt = np.array(c.findtext('maxPt').split()).astype(np.float)
            
            normals = getNormal(minPt, maxPt) # shading 작업 시 필요한 각 면의 normal vec, 매번 계산하면 overhead 크므로, 미리 계산

            ref = '' # 내부 변수를 사용하기 위해서 선언
            for child in c:
                if(child.tag == 'shader'): # 태그 shader, center, radius중에서 shader
                    ref = child.get('ref') # shader와 비교할 sphere의 ref 값
            
            for sh in root.findall('shader'):
                if(sh.get('name') == ref): # 현재 sphere의 shader ref와 동일한 것을 바로 찾음
                    shaderType = sh.get('type')
                    diffuseColor = np.array(sh.findtext('diffuseColor').split()).astype(np.float) 
                    if(sh.findtext('specularColor')):
                        specularColor = np.array(sh.findtext('specularColor').split()).astype(np.float)
                    if(sh.findtext('exponent')):
                        exponent = np.array(sh.findtext('exponent').split()).astype(np.float)                    
                    shader = Shader(shaderType, diffuseColor, specularColor, exponent) #shader 저장
                    surfaces.append(Box(minPt, maxPt, shader, normals)) #shader를 따로 저장하지 않고, surface에 편입하여 shader name 필요없음
    
    # light: xml -> class
    for c in root.findall('light'):
        position = np.array(c.findtext('position').split()).astype(np.float)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float)
        lights.append(Light(position, intensity))    
    

    # Create an empty image
    channels=3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:]=0
    
    w = getUnitV(-view.viewDir) # 역방향
    u = getUnitV(np.cross(view.viewUp, w)) # image plane 기준 x (좌 -> 우)
    v = getUnitV(np.cross(u, w)) #image plane 기준 y (상 -> 하)
    
    xPerPix = view.viewWidth / imgSize[0] # width per pixel
    yPerPix = view.viewHeight / imgSize[1] # height per pixel
    
    # starting vector, img plane에서 절반씩 좌상단으로 이동해 viewPoint로부터 img Plane 기준 (0,0) 좌표로 이동
    sVec = -w * view.projDistance + (-u * (view.viewWidth/2 + 1/2 * xPerPix)) + (-v * (view.viewHeight/2 + 1/2 * yPerPix))

    # pixel 순회하면서 하나씩 그리기 
    for x in np.arange(imgSize[0]):
        for y in np.arange(imgSize[1]):
            ray = sVec + u * x * xPerPix + v * y * yPerPix # 각 pixel 당 ray vec, 크기 무관
            hitSurface, t = intersect(surfaces, ray, view.viewPoint)
            img[y][x] = shade(surfaces, view, ray, lights, hitSurface, t)

    rawimg = Image.fromarray(img, 'RGB')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__=="__main__":
    main()

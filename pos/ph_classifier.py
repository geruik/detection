"""
燕进提供的原代码
"""

import math
import numpy as np

class PH_Classifier:
    def __init__(self):

        #人头检测用到的二次函数拟合
        self.thd = 0.8
        points = [(140, self.thd), (120, self.thd/2), (115, 0), (110, self.thd/2), (100, self.thd)]
        pointe = [(120, self.thd), (115, self.thd/2), (110, 0), (105, self.thd/2), (100, self.thd)]
        self.hs = self.WeightbyQuadratic(points)
        self.he = self.WeightbyQuadratic(pointe)

    def classify(self, keypoints, boxes):
        Human_Angle = 25 
        pose_score = {'Stand': 0.0, 'Fall': 0.0, 'Sit': 0.0, 'other': 0.0}
        head_score = {'HeadUp': 0.0, 'HeadDown': 0.0, 'unknown': 0.0}
        keypoint_indices = [5, 6, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4]
        keypoints_list = [keypoints[index][:2] for index in keypoint_indices]
        LShoulder, RShoulder, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle, Nose, LEye, REye, LEar, REar = keypoints_list
        
        CShoulder = self.CenterPoint(LShoulder, RShoulder)
        CHip = self.CenterPoint(LHip, RHip)
        CKnee = self.CenterPoint(LKnee, RKnee)
        CAnkle = self.CenterPoint(LAnkle, RAnkle)
        up_human_angle = self.Anglebyline([CHip,CShoulder],[[0, 0], [10, 0]])  #上身体中心线与水平线夹角
        down_human_angle = self.Anglebyline([CHip,CAnkle],[[0, 0], [10, 0]])  #下半身与水平线夹角
        aspect_ratio = self.aspectRatio(boxes)  #计算检测区域宽高比
        Hip_angle = self.Anglebypoint(CShoulder,CHip,CKnee)  #肩部胯部膝盖夹角
        Knee_angle = self.Anglebypoint(CHip,CKnee,CAnkle)  #胯部膝盖小腿夹角

        head_points = [Nose, LEye, REye, LEar, REar]        
        head_count = sum(1 for point in head_points if all(point))
        if head_count == 0:
            head_score['unknown'] += 3.0
            
        if head_count == 1:
            if all(Nose):
                head_score['HeadDown'] += 1.0
            else:
                head_score['unknown'] += 3.0

        # if head_count == 5:  #摄像头直拍才用上，俯拍就不要用。
        #     Ear_angle = self.Anglebypoint(RShoulder, REar, REye)
        #     if Ear_angle < 100:
        #         head_score['HeadDown'] += 0.8
        #     else:
        #         head_score['HeadUp'] += 0.8
            
      
        if (all(Nose) and all(LEye) and all(REye)):
            if head_count >= 4:
                head_score['HeadUp'] += 0.4
            else:
                head_score['HeadDown'] += 0.4

            # 检查鼻子或眼睛是否高于肩线投影

            if Nose[1] > LShoulder[1] or Nose[1] > RShoulder[1]:
                head_score['HeadDown'] += 0.8
            else:
                head_score['HeadUp'] += 0.2

        
        if head_count == 2 and all(LEar) and all(REar):
            head_score['HeadUp'] += 0.4

        #二次函数拟合来计算
        if (head_count == 1 and all(LEar)) or \
           (head_count == 2 and all(LEar) and all(LEye)) or \
           (head_count == 3 and all(LEar) and all(LEye) and all(Nose)):
            
            Shoulder_angle = self.Anglebypoint(LEar, LShoulder, LHip)
            weight_score = self.QuadraticCall(Shoulder_angle, self.hs[0], self.hs[1], self.hs[2])
            if Shoulder_angle > 115:
                head_score['HeadUp'] += weight_score
            else:
                head_score['HeadDown'] += weight_score
        
            if head_count >= 2 and all(LEar) and all(LEye):
                Ear_angle = self.Anglebypoint(LShoulder, LEar, LEye)
                weight_score = self.QuadraticCall(Ear_angle, self.he[0], self.he[1], self.he[2])
                if Ear_angle > 110:
                    head_score['HeadUp'] += weight_score
                else:
                    head_score['HeadDown'] += weight_score
                
        if (head_count == 1 and all(REar)) or \
           (head_count == 2 and all(REar) and all(REye)) or \
           (head_count == 3 and all(REar) and all(REye) and all(Nose)):
            
            Shoulder_angle = self.Anglebypoint(REar, RShoulder, RHip)
            weight_score = self.QuadraticCall(Shoulder_angle, self.hs[0], self.hs[1], self.hs[2])
            if Shoulder_angle > 115:
                head_score['HeadUp'] += weight_score
            else:
                head_score['HeadDown'] += weight_score
        
            if head_count >= 2 and all(REar) and all(REye):
                Ear_angle = self.Anglebypoint(RShoulder, REar, REye)
                weight_score = self.QuadraticCall(Ear_angle, self.he[0], self.he[1], self.he[2])
                if Ear_angle > 110:
                    head_score['HeadUp'] += weight_score
                else:
                    head_score['HeadDown'] += weight_score

        
        self.OTHERPOSE = 0
        otherkeypoint_indices = [5, 6, 11, 12, 13, 14, 15, 16]
        for index in keypoint_indices:
            if sum(keypoints[index][:2]) == 0:
                self.OTHERPOSE += 1
        if self.OTHERPOSE>=5: pose_score['other'] += 5.6
        
        #判断Shoulder、Hip、Knee是否被检测到
        if CKnee[0]== 0 and CKnee[1]== 0 and CHip[0]==0 and CHip[1] == 0:
            pose_score['Sit'] += 0.69
            pose_score['Fall'] += -0.8*2
            pose_score['Stand'] += -0.8*2

        elif CShoulder[1] == 0 and CShoulder[0] == 0 and CHip[0]==0 and CHip[1] == 0:
            pose_score['Sit'] += -0.8 * 2
            pose_score['Fall'] += -0.8 * 2
            pose_score['Stand'] += 0.69

        #身体中心线与水平线夹角
        if -180 <= up_human_angle <= Human_Angle:
            pose_score['Fall'] += 1.6
        elif 90 - Human_Angle <= up_human_angle <= 90 + Human_Angle:
            pose_score['Sit'] += 0.8
            pose_score['Stand'] += 0.8
        elif 90 + Human_Angle <= up_human_angle <= 90 + (2 * Human_Angle):
            pose_score['Sit'] += 0.3
            pose_score['Fall'] += 0.3
        elif 90 + (2 * Human_Angle) < up_human_angle <= 180:
            pose_score['Fall'] += 1.6
        else:
            pose_score['Fall'] += 0.8 * ((90 - abs(up_human_angle)) / 90)
        if all(LAnkle) and all(RAnkle) and all(LHip) and all(RHip):
            if (-Human_Angle <= down_human_angle < Human_Angle or
                down_human_angle > 180 - Human_Angle or
                down_human_angle < Human_Angle - 180):
                pose_score['Fall'] += 1.0

            
        #通过检测框的长宽比来判断，反正就那样吧。。。。。
        if aspect_ratio[0] < 0.5 and 65 <= up_human_angle < 135:
            pose_score['Stand'] += 0.8
        elif 0.6 <= aspect_ratio[0] < 1.6 and 65 <= up_human_angle < 135:
            pose_score['Sit'] += 0.8
        elif aspect_ratio[0] >= 1.6:
            pose_score['Fall'] += 0.8


        #
        if 45 < Hip_angle < 135 and 45 < up_human_angle <135:
            pose_score['Sit'] += 0.6
            pose_score['Stand'] += 0.3
        elif Hip_angle > 135 and 45 < up_human_angle <135:
            pose_score['Stand'] += 0.3
        elif Hip_angle < 135 and up_human_angle <45:
            pose_score['Fall'] += 0.4
        else:
            pose_score['Fall'] += 0.2
            pose_score['Stand'] += 0.2

        if 70 < Knee_angle < 135 and 45 < up_human_angle <135:
            pose_score['Sit'] += 0.6
            pose_score['Stand'] += -0.035
        elif Knee_angle> 135 and 45 < up_human_angle <135:
            pose_score['Stand'] += 0.3
            pose_score['Fall'] += 0.5
        else:
            pose_score['Fall'] += 0.3
            pose_score['Stand'] += 0.3

        
        score_max, status_max = max(zip(pose_score.values(), pose_score.keys()))
        head_score_max, head_status_max = max(zip(head_score.values(), head_score.keys()))
        
        if status_max == 'Fall' or status_max == 'other':
            status_max = 'unknown'
        
        return status_max, head_status_max

    def Anglebyline(self, l1, l2):
        try:
            vector1 = (l1[1][0] - l1[0][0], l1[1][1] - l1[0][1])
            vector2 = (l2[1][0] - l2[0][0], l2[1][1] - l2[0][1])
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
            cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
            if cross_product < 0:
                angle_rad = -angle_rad
            angle_deg = math.degrees(angle_rad)
            return angle_deg
        except ZeroDivisionError:
            print("线段不存在，无法计算两条直线夹角。")
            return 0

    def aspectRatio(self,boxes):
        try:
            x1, y1, x2, y2 = boxes
            W = x2-x1
            H = y2-y1
            aspect_ratio = W / H
            return [aspect_ratio,W,H]
        except ZeroDivisionError:
            print("高度不能为0，无法计算宽高比。")

    def Anglebypoint(self, p1, p2, p3): 
        try:
            a = math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1]))
            b = math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1]))
            c = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
            B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
            return B
        except ZeroDivisionError:
            return 0

    def CenterPoint(self, p1, p2):
        return [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2]
        
    def WeightbyQuadratic(self,points):
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, b, c
    
    def QuadraticCall(self,x, a, b, c):
        weight = a * x**2 + b * x + c
        if weight > self.thd:
            weight == self.thd
        return weight


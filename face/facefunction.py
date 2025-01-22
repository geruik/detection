import time
import uuid
import cv2
from loguru import logger
import pymysql
import numpy as np
import insightface
from sklearn import preprocessing
import sqlite3
import base64
import json
import config
from skimage import transform as trans

from ultralytics import YOLO


class FaceRecognition:
    def __init__(self, process_share, model_load=True):
        self.model = None
        self.emotion_model = None
        self.threshold = 1.45 #用于后续register
        self.stranger_threshold = 1.6 #用于判断是否进行陌生人注册人脸
        self.faces_common = list() # 初始化通用人脸特征库list
        self.faces_visitor = list() # 初始化游客人脸库list
        self.faces_stranger = list() # 初始化陌生人库list
        self.process_share = process_share
        self.update_time = int(time.time())
        self.maplist = {0: 'Angry', 1: 'Fearful', 2: 'Happy', 3: 'Sad', 4: 'Netural'}
        self.catch_emotion = [0,1,2,3]
        self.model_load = model_load
        if self.model_load:
            self.load_models()
        #加载数据库
        self.load_faces()
        
    # 模型加载
    def load_models(self):
        logger.info("开始加载模型")
        self.model = insightface.app.FaceAnalysis(root='./default_models', #models文件夹所在，问就是app.FaceAnalysis这样定义的,最好用绝对地址
                                                  name='face_model', #model里面的模型组，cpu和gpu都可以用，数字越高能力越强占用越多
                                                  allowed_modules=None, 
                                                  providers=['CUDAExecutionProvider']) #CPUExecutionProvider
        #ctx_id=gpu_id=0 选择第0个GPU， 正数为GPU的ID，负数为使用CPU，检测阈值，和输入检测的size
        self.model.prepare(ctx_id=0, det_thresh=0.50, det_size=(640, 640))
        self.emotion_model = YOLO("default_models/emotion3v.pt")
        self.model_load = True

    def update_face_time(self):
        self.update_time = int(time.time())

    # 加载人脸库中的人脸
    def load_faces(self):
        logger.info("开始加载人脸库")
        conn = create_connection()
        if conn:
            self.faces_common.clear()        #清空原本存储的人脸数据
            self.faces_visitor.clear()
            self.faces_stranger.clear()
            cur = conn.cursor()
            # 获取人脸库中的通用人脸信息
            cur.execute("SELECT id, user_name, feature, age, gender, type FROM faces")
            rows = cur.fetchall()
            for row in rows:
                embedding = np.array(json.loads(row[2])).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                if "common" == row[5]:
                    self.faces_common.append({
                        "id": row[0],
                        "user_name": row[1],
                        "feature": embedding,
                        "age": row[3],
                        "gender": row[4],
                        "type": row[5],
                    })
                elif "visitor" == row[5]:
                    self.faces_visitor.append({
                        "id": row[0],
                        "user_name": row[1],
                        "feature": embedding,
                        "age": row[3],
                        "gender": row[4],
                        "type": row[5],
                    })
                elif "stranger" == row[5]:
                    self.faces_stranger.append({
                        "id": row[0],
                        "user_name": row[1],
                        "feature": embedding,
                        "age": row[3],
                        "gender": row[4],
                        "type": row[5],
                    })
            close_connection(conn)

    def get_face_img(self, id):
        conn = create_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT face_image FROM faces where id = %s", (id,))
            row = cur.fetchone()
            if row is None:
                return None
            else:
                return cv2.imdecode(np.frombuffer(base64.b64decode(row[0].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            close_connection(conn)
            
    # 检测
    def detect(self, image):
        if not self.model_load:
            self.load_models()
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            if hasattr(face, 'kps') and face.kps is not None:
                result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()
            if hasattr(face, 'pose') and face.pose is not None:
                result["pose"] = np.array(face.pose).astype(np.int32).tolist()
            if hasattr(face, 'age') and face.age is not None:
                result["age"] = face.age
            if hasattr(face, 'gender') and face.gender is not None:
                gender = '男' if face.gender != 0 else '女'
                result["gender"] = gender
            if hasattr(face, 'embedding') and face.embedding is not None:
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                result["embedding"] = embedding
            results.append(result)
        
        return results

    #注册
    def register(self, user_name, face_embedding, face_image, age, gender, type):
        conn = create_connection()
        if conn:
            cur = conn.cursor()
            # cur.execute('SELECT * FROM faces WHERE user_name = ? and type = ?', (user_name,type))
            # if cur.fetchone() is not None:
            #     return '该用户已存在'
            age = age if age is not None else ''
            gender = gender if gender is not None else ''

            embedding_json = json.dumps(face_embedding.tolist()) 
            
            cur.execute('''
            INSERT INTO faces (user_name, feature, face_image, age, gender, type)
            VALUES (%s, %s, %s, %s, %s, %s)
            ''', (user_name, embedding_json, face_image, age, gender, type))
            conn.commit()
            self.process_share.value = int(time.time())
            close_connection(conn)
        return "success"

    # 识别
    def recognition(self, image, det_score = 0.7):
        if not self.model_load:
            self.load_models()
        current_time = int(time.time())
        if  (current_time - self.update_time) >= 1800:
            self.load_faces()
            self.update_time = current_time
        faces = self.model.get(image)
        results = list()
        for face in faces:
        #识别不涉及模型，只是一些一些计算，如果要用概率，可以用这个1 / (1 + math.exp(5 * (x - 1))) 或者自己定义逻辑函数
            if hasattr(face, 'det_score') and face.det_score is not None:
                if face.det_score < det_score:
                    continue

            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            min_dist = float('inf')   #也可以不用死板用inf
            matched_face = None
            for com_face in self.faces_common:
                dist = self.compare(embedding, com_face["feature"])
                if dist < self.threshold and dist < min_dist:
                    min_dist = dist
                    matched_face = com_face
            if matched_face:
                user_name = matched_face["user_name"]
            results.append({
                "id": None if matched_face is None else matched_face["id"],
                "user_name": user_name,
                "dist": min_dist,
                "bbox": np.maximum(np.array(face.bbox), 0).astype(np.int32).tolist()
            })
        return results
    
    # 识别，进阶版
    def recognition_advance(self, image, det_score = 0.7, expression_conf = 0.6, register = True, emotion_detect = False):
        if not self.model_load:
            self.load_models()
        current_time = int(time.time())
        if  (current_time - self.update_time) >= 1800:
            self.load_faces()
            self.update_time = current_time
        faces = self.model.get(image)
        detect_results = list()
        for face in faces:
            if hasattr(face, 'det_score') and face.det_score is not None:
                if face.det_score < det_score:
                    continue
            #情绪检测
            emotion = None
            if emotion_detect:
                aligned_face, _ = self.align_face(image, (112, 112), np.array(face['kps']))
                results = self.emotion_model(aligned_face, save=False)
                for r in results:
                    if r.probs.top1conf.item() < expression_conf:
                        continue
                    expression_id = r.probs.top1
                    if expression_id in self.catch_emotion:
                        emotion = self.maplist[expression_id]
            #识别不涉及模型，只是一些一些计算，如果要用概率，可以用这个1 / (1 + math.exp(5 * (x - 1))) 或者自己定义逻辑函数
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            type = "stranger"
            min_dist = float('inf')   #也可以不用死板用inf
            temp_dist = float('inf')
            similarity = 0.0
            matched_face = None
            for com_face in self.faces_visitor:
                dist = self.compare(embedding, com_face["feature"])
                if dist < temp_dist:
                    temp_dist = dist
                if dist < self.threshold and dist < min_dist:
                    min_dist = dist
                    matched_face = com_face
            if matched_face:
                user_name = matched_face["user_name"]
                type = "visitor"
                similarity = float(1 - np.exp(-3.0 * (1.5- min_dist)))
            else:
                for com_face in self.faces_common:
                    dist = self.compare(embedding, com_face["feature"])
                    if dist < temp_dist:
                        temp_dist = dist
                    if dist < self.threshold and dist < min_dist:
                        min_dist = dist
                        matched_face = com_face
                if matched_face:
                    user_name = matched_face["user_name"]
                    type = "common"
                    similarity = float(1 - np.exp(-3.0 * (1.5- min_dist)))
                else:
                    type = "stranger"
                    for com_face in self.faces_stranger:
                        dist = self.compare(embedding, com_face["feature"])
                        if dist < self.threshold and dist < min_dist:
                            min_dist = dist
                            matched_face = com_face
                    if matched_face:
                        user_name = matched_face["user_name"]
                    else:
                        if register:
                            if temp_dist < self.stranger_threshold:
                                logger.info(f"识别到距离值未到注册程度的人脸，距离值: {temp_dist}")
                                continue
                            # 注册为陌生人人脸
                            user_name = str(uuid.uuid4())[:8]
                            bbox = np.array(face.bbox).astype(np.int32).tolist()
                            x1, y1, x2, y2 = bbox
                            # 计算扩大后的边界框坐标
                            margin_x = int((x2 - x1) * 0.5)  # 计算x方向扩大的0.5倍范围
                            margin_y = int((y2 - y1) * 0.5)  # 计算y方向扩大的0.5倍范围
                            # 确保新的坐标在图像范围内
                            new_x1 = max(0, x1 - margin_x)
                            new_y1 = max(0, y1 - margin_y)
                            new_x2 = min(image.shape[1], x2 + margin_x)  # frame.shape[1] 是图像的宽度
                            new_y2 = min(image.shape[0], y2 + margin_y)  # frame.shape[0] 是图像的高度
                            # 使用新的边界框坐标来截取 face_img
                            face_img = image[new_y1:new_y2, new_x1:new_x2]
                            _, encoded_img = cv2.imencode(".jpg", face_img)
                            encoded_img_base64 = base64.b64encode(encoded_img).decode("utf-8")
                            face_image_base64 = (f"data:image/jpg;base64,{encoded_img_base64}")
                            self.register(
                                user_name,
                                embedding,
                                face_image_base64,
                                None,
                                None,
                                "stranger"
                            )
            detect_results.append({
                "id": None if matched_face is None else matched_face["id"],
                "user_name": user_name,
                "dist": min_dist,
                "bbox": np.maximum(np.array(face.bbox), 0).astype(np.int32).tolist(),
                "type": type,
                "emotion": emotion,
                "similarity": similarity,
            })
        return detect_results

    @staticmethod  #主要是通过np.subtract 计算向量距离，可以选择别的方式，但是threshold和dist需要根据计算方式变化。
    def compare(embedding1, embedding2):
        diff = np.subtract(embedding1, embedding2)
        dist = np.sum(np.square(diff), 1)
        return dist
            
    def delete(self,id_str,type):
        conn = create_connection()
        if conn: 
            try:
                cur = conn.cursor()
                id_list = id_str.replace("'", "").replace(" ", "").split(",")  # 假设 id_str 是用逗号分隔的字符串
                placeholders = ', '.join(['%s'] * len(id_list))  # 正确生成占位符
                
                sql = f"DELETE FROM faces WHERE user_name IN ({placeholders}) AND type = %s"
                cur.execute(sql, (*id_list, type))  # 解包参数
                
                conn.commit()
                logger.info(f"成功删除用户id为{id_str}、用户类型为{type}的人脸数据")
                self.process_share.value = int(time.time())
            except Exception as e:
                logger.error(f"删除操作失败: {e}")
            finally:
                close_connection(conn)

    def update_user_name(self,original_name,new_name):
        conn = create_connection()
        if conn:
            cur = conn.cursor()
            sql = f"UPDATE faces SET user_name = %s WHERE user_name = %s  AND type = 'stranger'"                
            cur.execute(sql, (new_name, original_name))
            conn.commit()

            logger.info(f"成功修改陌生人原id:{original_name}为新id:{new_name}")
            self.process_share.value = int(time.time())
            close_connection(conn)

    #新定义的对齐人脸
    def align_face(self,image, size, lmks):
        dst_w = size[1]
        dst_h = size[0]
        base_w = 96
        base_h = 112
        assert (dst_w >= base_w)
        assert (dst_h >= base_h)
        base_lmk = [
            30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
            71.7366, 33.5493, 92.3655, 62.7299, 92.2041
        ]
        dst_lmk = np.array(base_lmk).reshape((5, 2)).astype(np.float32)
        if dst_w != base_w:
            slide = (dst_w - base_w) / 2
            dst_lmk[:, 0] += slide
        if dst_h != base_h:
            slide = (dst_h - base_h) / 2
            dst_lmk[:, 1] += slide
        src_lmk = lmks
        tform = trans.SimilarityTransform()
        tform.estimate(src_lmk, dst_lmk)
        t = tform.params[0:2, :]
        assert (image.shape[2] == 3)
        dst_image = cv2.warpAffine(image.copy(), t, (dst_w, dst_h))
        dst_pts = self.GetAffinePoints(src_lmk, t)
        return dst_image, dst_pts

    def GetAffinePoints(self,pts_in, trans):
        pts_out = pts_in.copy()
        assert (pts_in.shape[1] == 2)
        for k in range(pts_in.shape[0]):
            pts_out[k, 0] = pts_in[k, 0] * trans[0, 0] + pts_in[k, 1] * trans[0, 1] + trans[0, 2]
            pts_out[k, 1] = pts_in[k, 0] * trans[1, 0] + pts_in[k, 1] * trans[1, 1] + trans[1, 2]
        return pts_out
    


def create_connection():
    try:
        # 建立连接
        connection = pymysql.connect(
            host=config.mysql["host"],  # MySQL服务器地址
            database=config.mysql["database"],  # 数据库名称
            user=config.mysql["user"],  # 用户名
            password=config.mysql["password"]  # 密码
        )
        
        return connection

    except Exception as e:
        logger.error(f"连接人脸数据库失败，失败原因: {e}")
        return None
    
def close_connection(connection):
    connection.close()

        

    
DATABASE_FILE = 'face/face.db'
__initialized = False
recognize_object = None

def initialize(process_share):
    """初始化本模块"""
    global __initialized  # 声明 __initialized 为全局变量
    if __initialized is False:
        # 执行初始化操作
        __do_init__(process_share)
    __initialized = True

def __do_init__(process_share):
    logger.info(f"模块 {__name__} 执行__do_init__...")
    try:
        _create_database()
        getRecognizaObject(process_share,False)
    except Exception as e:
        logger.error(f"初始化模块{__name__}失败，异常信息{e}")

def _create_database():
    conn = create_connection()
    if conn: 
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS `faces` (
                `id` bigint(32) NOT NULL AUTO_INCREMENT,
                `user_name` varchar(100) NOT NULL,
                `feature` text NOT NULL,
                `face_image` longtext NOT NULL,
                `age` varchar(20) DEFAULT NULL,
                `gender` varchar(20) DEFAULT NULL,
                `type` varchar(20) DEFAULT NULL,
                PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        conn.commit()
        close_connection(conn)

def getRecognizaObject(process_share, model_load=True):
    global recognize_object
    if recognize_object is None:
        recognize_object = FaceRecognition(process_share,model_load)
    return recognize_object

"""
            CREATE TABLE IF NOT EXISTS `faces_update` (
                `id` bigint(32) NOT NULL AUTO_INCREMENT,
                `updated_time` bigint(32) NOT NULL,
                PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

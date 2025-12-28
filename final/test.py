import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading
import time
import random
import os
import warnings

warnings.filterwarnings("ignore")

# ===================== 资源路径配置 =====================
A_F = "sound"  # 音频文件夹路径
a_blk = os.path.join(A_F, "block")  # 常规攻击音效文件夹
a_lst = ["1.mp3", "2.mp3", "3.mp3"]  # 攻击音效文件列表
a_dth = "death.mp3"  # 死亡音效
a_atk = "attack2.mp3"  # 攻击音效
a_dan = "danger.mp3"  # 危险提示音
a_lit = "lightning.mp3"  # 闪电音效
a_exc = "excution.mp3"  # 处决音效
a_man = "aaa.mp3"  # 终局音效
a_str = "start.mp3"  # 开始界面背景音乐
a_end = "end.mp3"  # 结束音效
a_tt = "tt.mp3"  # 教程背景音乐

a_cd = 1  # 音效冷却时间(秒)
b_files = ['0.png', '1.png', '2.png', '3.png']  # 背景图片序列
b_stop = "images/stop.png"  # 停止状态背景图
b_exc = "images/excution.png"  # 处决状态背景图
b_thx = "images/thanks.png"  # 感谢界面图
START_IMG = "images/start.png"  # 开始界面图

s_dt = 3.5  # 倒计时总时长
s_ld = 1.8  # 倒计时显示时长
b_seq = [1, 0, 2, 0, 1, 0, 2, 0, 2, 0, 1, 0, 3, 0]  # 游戏背景序列(1右2左3终局)
b_iv = 0.6  # 背景切换间隔
b_lp = True  # 是否循环背景序列
b_it = 3  # 初始等待时间
w_at = 0.4  # 预警时间
b3_et = 1.5  # 终局前预警时间
b3_dt = 3  # 终局反应时间

l_iv = 0.4  # 闪电生成间隔
l_du = 0.3  # 闪电显示时长
l_cs = 5  # 闪电连续击中次数阈值
l_ms = 25  # 主闪电段数
l_mb = 8  # 分支数量上限
l_jn = 40  # 闪电抖动幅度
l_jx = 80  # 闪电抖动幅度上限
l_cn = 5  # 闪电数量下限
l_cx = 8  # 闪电数量上限
l_cc = (0, 255, 255)  # 闪电主颜色
l_gc = (255, 255, 255)  # 闪电辉光颜色
l_cw = 3  # 闪电主线宽度
l_gw = 10  # 闪电辉光宽度
l_br = 8  # 闪电模糊半径
l_gi = 0.6  # 闪电透明度

b3_tr = 0.3  # 终局线位置(顶部)
b3_lr = 0.2  # 终局线位置(左侧)
b3_rr = 0.8  # 终局线位置(右侧)
b3_th = 10  # 终局线判定阈值
b3_ic = (0, 0, 255)  # 终局线初始颜色
b3_cc = (0, 255, 0)  # 终局线完成颜色
b3_w = 1  # 终局线宽度

g_th = 0.2  # 抓取检测阈值
g_fn = 5  # 抓取检测帧数
g_iv = 0.1  # 抓取检测间隔

k_path = "images/weapon/1.3.png"  # 武器图片路径
k_sf = 0.7  # 武器缩放系数

h_paths = ["images/hp0.png", "images/hp1.png", "images/hp2.png", "images/hp3.png"]  # HP条图片路径
h_sf = 0.018  # HP条缩放系数
h_pos = "bottom_left"  # HP条位置
h_xo = 15  # HP条X偏移
h_yo = 20  # HP条Y偏移

d_path = "images/cai_result.png"  # 死亡图片路径
d_dt = 2.0  # 死亡显示时长

p_paths = ["images/posture1.png", "images/posture2.png", "images/posture3.png", "images/posture4.png",
           "images/posture5.png", "images/posture6.png", "images/posture7.png", "images/posture8.png"]  # 姿势图片路径
p_sf = 0.015  # 姿势缩放系数
p_pos = "top_center"  # 姿势位置
p_xo = 0  # 姿势X偏移
p_yo = 20  # 姿势Y偏移
p_dt = 2.0  # 姿势显示时长

r_path = "images/red.png"  # 红屏提示路径
r_sf = 0.3  # 红屏缩放系数
r_tr = 0.6  # 红屏透明度
r_br = 15  # 红屏模糊半径
r_cr = 0.9  # 红屏裁剪半径

dan_path = "images/danger_result.png"  # 危险提示图路径
dan_sf = 0.5  # 危险提示缩放
dan_tr = 1.0  # 危险提示透明度
dan_fd = 0.3  # 危险提示淡入时间
dan_ss = 0.8  # 危险提示起始缩放
dan_se = 1.2  # 危险提示结束缩放

cam_w = 400  # 摄像头宽度
cam_h = 420  # 摄像头高度
cdw = 120  # 摄像头显示宽度
cdh = 90  # 摄像头显示高度
ds = 2  # 显示缩放系数

l_lr = 0.3  # 左线位置比例
l_rr = 0.7  # 右线位置比例
l_th = 5  # 线判定阈值
w_al = 0.2  # 警告透明度

mdc = 0.5  # MediaPipe检测置信度
mtc = 0.5  # MediaPipe跟踪置信度
mnh = 2  # 最大手数

py_fr = 44100  # 音频采样率
py_ch = 2  # 音频通道数
py_bf = 4096  # 音频缓冲区大小

PINCH_TH = 15  # 捏合判定阈值

TUTORIAL_FOLDER = "images/tutorial"  # 教程图片文件夹
TUTORIAL_BACK_BTN = (10, 10, 80, 40)  # 教程返回按钮区域
TUTORIAL_SWIPE_THRESH = 30  # 教程滑动阈值
TUTORIAL_EXIT_BTN = (160, 300, 80, 40)  # 教程退出按钮区域
TUTORIAL_COOLDOWN = 1.0  # 教程操作冷却

# ===================== 初始化音频系统 =====================
pygame.mixer.init(frequency=py_fr, size=-16, channels=py_ch, buffer=py_bf)

# ===== 人脸检测初始化(用于excution模式人脸替换) =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FACE_RATIO = (0.55, 0, 0.285, 0.35)  # 人脸在excution.png上的显示区域比例(x,y,w,h)

# ===================== 加载音频资源 =====================
# 加载所有攻击音效
a_snds = []
for f in [os.path.join(a_blk, f) for f in a_lst]:
    a_snds.append(pygame.mixer.Sound(f) if os.path.exists(f) else None)
a_ok = any(s is not None for s in a_snds)

# 加载其他音效
d_snd = pygame.mixer.Sound(os.path.join(A_F, a_dth)) if os.path.exists(os.path.join(A_F, a_dth)) else None
d_ok = d_snd is not None
atk_snd = pygame.mixer.Sound(os.path.join(A_F, a_atk)) if os.path.exists(os.path.join(A_F, a_atk)) else None
atk_ok = atk_snd is not None
dan_snd = pygame.mixer.Sound(os.path.join(A_F, a_dan)) if os.path.exists(os.path.join(A_F, a_dan)) else None
dan_ok = dan_snd is not None
lit_snd = pygame.mixer.Sound(os.path.join(A_F, a_lit)) if os.path.exists(os.path.join(A_F, a_lit)) else None
lit_ok = lit_snd is not None
exc_snd = pygame.mixer.Sound(os.path.join(A_F, a_exc)) if os.path.exists(os.path.join(A_F, a_exc)) else None
exc_ok = exc_snd is not None
m_snd = pygame.mixer.Sound(os.path.join(A_F, a_man)) if os.path.exists(os.path.join(A_F, a_man)) else None
m_ok = m_snd is not None
strt_snd = pygame.mixer.Sound(os.path.join(A_F, a_str)) if os.path.exists(os.path.join(A_F, a_str)) else None
strt_ok = strt_snd is not None
end_snd = pygame.mixer.Sound(os.path.join(A_F, a_end)) if os.path.exists(os.path.join(A_F, a_end)) else None
end_ok = end_snd is not None

# ===================== 加载图片资源 =====================
# 加载停止状态背景图
s_bg = cv2.imread(b_stop)
s_bg_ok = s_bg is not None
if s_bg_ok:
    s_bg = cv2.resize(s_bg, (400, 420))

# 加载处决状态背景图
e_bg = cv2.imread(b_exc)
e_bg_ok = e_bg is not None
if e_bg_ok:
    e_bg = cv2.resize(e_bg, (400, 420))

# 加载感谢界面图
t_img = cv2.imread(b_thx, cv2.IMREAD_UNCHANGED)
t_ok = t_img is not None
if t_ok:
    # 处理透明度通道
    if t_img.shape[2] == 4:
        t_msk = t_img[:, :, 3]
        t_img = t_img[:, :, :3]
    else:
        t_hsv = cv2.cvtColor(t_img, cv2.COLOR_BGR2HSV)
        t_msk = cv2.bitwise_not(cv2.inRange(t_hsv, (35, 40, 40), (85, 255, 255)))
    t_img = cv2.resize(t_img, (400, 420))
    t_msk = cv2.resize(t_msk, (400, 420))

# 加载开始界面图
start_bg = cv2.imread(START_IMG)
start_bg_ok = start_bg is not None
if start_bg_ok:
    start_bg = cv2.resize(start_bg, (400, 420))

# ===================== 初始化MediaPipe =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=mnh,
                       min_detection_confidence=mdc, min_tracking_confidence=mtc)
face_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ===================== 加载游戏背景图 =====================
dw, dh = 400, 420
bgs = []
for f in b_files:
    if os.path.exists(f):
        tmp = cv2.imread(f)
        if tmp is not None:
            tmp = cv2.resize(tmp, (dw, dh))
            bgs.append(tmp)
            continue
    bgs.append(np.zeros((dh, dw, 3), dtype=np.uint8))
if s_bg_ok:
    bgs.append(s_bg)
else:
    bgs.append(np.zeros((dh, dw, 3), dtype=np.uint8))
bh, bw = bgs[0].shape[:2]

# 终局线位置计算
b3_ty = int(bh * b3_tr)
b3_lx = int(bw * b3_lr)
b3_rx = int(bw * b3_rr)

# ===================== 加载武器图片 =====================
k_ok = False
if os.path.exists(k_path):
    k_img = cv2.imread(k_path, cv2.IMREAD_UNCHANGED)
    if k_img is not None:
        # 处理透明度
        if k_img.shape[2] == 4:
            k_msk = k_img[:, :, 3]
            k_img = k_img[:, :, :3]
        else:
            k_hsv = cv2.cvtColor(k_img, cv2.COLOR_BGR2HSV)
            k_msk = cv2.bitwise_not(cv2.inRange(k_hsv, (35, 40, 40), (85, 255, 255)))
        # 缩放武器图
        h, w = k_img.shape[:2]
        nh = int(bh * k_sf)
        nw = int(w * nh / h)
        k_img = cv2.resize(k_img, (nw, nh))
        k_msk = cv2.resize(k_msk, (nw, nh))
        k_img_f = cv2.flip(k_img, 1)  # 水平翻转(右手用)
        k_msk_f = cv2.flip(k_msk, 1)
        k_ok = True

# ===================== 加载HP条图片 =====================
h_imgs, h_msks = [], []
h_ok = False
chp = 3  # 当前HP
for p in h_paths:
    if os.path.exists(p):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 4:
                msk = img[:, :, 3]
                img = img[:, :, :3]
            else:
                msk = np.ones(img.shape[:2], dtype=np.uint8) * 255
            # 缩放HP条
            h, w = img.shape[:2]
            nh = int(bh * h_sf)
            nw = int(w * nh / h)
            img = cv2.resize(img, (nw, nh))
            msk = cv2.resize(msk, (nw, nh))
            h_imgs.append(img)
            h_msks.append(msk)
if h_imgs:
    h_ok = True
    hw, hh = h_imgs[0].shape[1], h_imgs[0].shape[0]

# ===================== 加载死亡图片 =====================
d_img = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
d_ok = d_img is not None
if d_ok:
    # 处理透明度
    if d_img.shape[2] == 4:
        d_msk = d_img[:, :, 3]
        d_img = d_img[:, :, :3]
    else:
        d_hsv = cv2.cvtColor(d_img, cv2.COLOR_BGR2HSV)
        d_msk = cv2.bitwise_not(cv2.inRange(d_hsv, (35, 40, 40), (85, 255, 255)))
    # 缩放死亡图
    dh0, dw0 = d_img.shape[:2]
    ndw = int(bw * 0.5)
    ndh = int(dh0 * ndw / dw0)
    d_img = cv2.resize(d_img, (ndw, ndh))
    d_msk = cv2.resize(d_msk, (ndw, ndh))

# ===================== 加载姿势图片 =====================
p_imgs, p_msks = [], []
p_ok = False
cp = 0  # 当前姿势索引
pst_shown = False  # 是否显示过姿势
pst_max = False  # 是否达到最大姿势
for p in p_paths:
    if os.path.exists(p):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 4:
                msk = img[:, :, 3]
                img = img[:, :, :3]
            else:
                msk = np.ones(img.shape[:2], dtype=np.uint8) * 255
            # 缩放姿势图
            h, w = img.shape[:2]
            nh = int(bh * p_sf)
            nw = int(w * nh / h)
            img = cv2.resize(img, (nw, nh))
            msk = cv2.resize(msk, (nw, nh))
            p_imgs.append(img)
            p_msks.append(msk)
if p_imgs:
    p_ok = True
    pw, ph = p_imgs[0].shape[1], p_imgs[0].shape[0]

# ===================== 加载红屏提示图片 =====================
r_img = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
r_ok = r_img is not None
if r_ok:
    # 处理透明度
    if r_img.shape[2] == 4:
        r_msk = r_img[:, :, 3]
        r_img = r_img[:, :, :3]
    else:
        r_hsv = cv2.cvtColor(r_img, cv2.COLOR_BGR2HSV)
        r_msk = cv2.bitwise_not(cv2.inRange(r_hsv, (35, 40, 40), (85, 255, 255)))
    # 缩放红屏图
    rh0, rw0 = r_img.shape[:2]
    nrw = int(bw * r_sf)
    nrh = int(rh0 * nrw / rw0)
    r_img = cv2.resize(r_img, (nrw, nrh))
    r_msk = cv2.resize(r_msk, (nrw, nrh))
    # 应用模糊效果
    if r_br > 0:
        br = r_br if r_br % 2 == 1 else r_br + 1
        r_img = cv2.GaussianBlur(r_img, (br, br), 0)
        r_msk = cv2.GaussianBlur(r_msk, (br, br), 0)
    # 创建圆形遮罩
    rc_msk = np.zeros((nrh, nrw), dtype=np.uint8)
    cx, cy = nrw // 2, nrh // 2
    cr = int(min(nrw, nrh) / 2 * r_cr)
    cv2.circle(rc_msk, (cx, cy), cr, 255, -1)
    r_msk = cv2.bitwise_and(r_msk, r_msk, mask=rc_msk)
    if r_br > 0:
        rc_msk = cv2.GaussianBlur(rc_msk, (br, br), 0)

# ===================== 加载危险提示图片 =====================
dan_img = cv2.imread(dan_path, cv2.IMREAD_UNCHANGED)
dan_ok = dan_img is not None
if dan_ok:
    # 处理透明度
    if dan_img.shape[2] == 4:
        dan_msk = dan_img[:, :, 3]
        dan_img = dan_img[:, :, :3]
    else:
        dan_hsv = cv2.cvtColor(dan_img, cv2.COLOR_BGR2HSV)
        dan_msk = cv2.bitwise_not(cv2.inRange(dan_hsv, (35, 40, 40), (85, 255, 255)))
    dan_h0, dan_w0 = dan_img.shape[:2]


# ===================== 核心功能函数 =====================
def gen_lit_path(s, e, ns=8, j=40):
    """
    生成闪电路径
    s: 起点坐标
    e: 终点坐标
    ns: 路径段数
    j: 抖动幅度
    """
    pts = [s]
    v = np.array(e) - np.array(s)
    for i in range(1, ns):
        t = i / ns
        bp = np.array(s) + v * t
        # 在中间段添加随机抖动
        if 0.2 < t < 0.8:
            n = np.array([-v[1], v[0]])
            nm = np.linalg.norm(n)
            if nm > 0:
                n = n / nm
                jr = l_jx - l_jn
                d = (random.random() * 2 - 1) * (l_jn + random.random() * jr) * (1 - abs(t - 0.5) * 2)
                bp = bp + n * d
        pts.append(tuple(bp.astype(int)))
    pts.append(e)
    return pts


def draw_lit(bg, paths, branches):
    """
    绘制闪电效果
    bg: 背景图
    paths: 主闪电路径列表
    branches: 分支路径列表
    """
    if not paths and not branches:
        return bg
    # 创建辉光层
    glow = np.zeros_like(bg, dtype=np.uint8)
    for p in paths:
        if len(p) >= 2:
            pts = np.array(p, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(glow, [pts], False, l_gc, thickness=l_gw, lineType=cv2.LINE_AA)
            cv2.polylines(glow, [pts], False, l_gc, thickness=l_gw + 4, lineType=cv2.LINE_AA)
    for b in branches:
        if len(b) >= 2:
            pts = np.array(b, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(glow, [pts], False, l_gc, thickness=l_gw // 2, lineType=cv2.LINE_AA)
            cv2.polylines(glow, [pts], False, l_gc, thickness=l_gw // 2 + 2, lineType=cv2.LINE_AA)
    # 应用模糊
    if l_br > 0:
        br = l_br if l_br % 2 == 1 else l_br + 1
        glow = cv2.GaussianBlur(glow, (br, br), 0)
    # 混合辉光
    res = cv2.addWeighted(bg, 1.0, glow, l_gi, 0)
    # 绘制主线
    for p in paths:
        if len(p) >= 2:
            pts = np.array(p, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(res, [pts], False, l_cc, thickness=l_cw, lineType=cv2.LINE_AA)
            cv2.polylines(res, [pts], False, (0, 200, 200), thickness=l_cw - 1, lineType=cv2.LINE_AA)
    for b in branches:
        if len(b) >= 2:
            pts = np.array(b, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(res, [pts], False, l_cc, thickness=l_cw // 2, lineType=cv2.LINE_AA)
            cv2.polylines(res, [pts], False, (0, 200, 200), thickness=l_cw // 2 - 1, lineType=cv2.LINE_AA)
    return res


def chk_grasp(lm, w, h):
    """
    检测是否抓取手势
    lm: 手部关键点
    w: 图像宽度
    h: 图像高度
    """
    tips = [8, 12, 16, 20]  # 指尖索引
    pips = [6, 10, 14, 18]  # 指关节索引
    t = lm.landmark[4]  # 拇指尖
    ip = lm.landmark[3]  # 拇指关节
    tb = t.x > ip.x  # 拇指是否弯曲
    bf = 0
    for ti, pi in zip(tips, pips):
        tp = lm.landmark[ti]
        pp = lm.landmark[pi]
        if tp.y > pp.y:  # 指尖在关节下方(弯曲)
            bf += 1
    return tb and bf == 4  # 拇指弯曲且四指都弯曲


def stop_all():
    """停止所有音频播放"""
    pygame.mixer.stop()


def play_lit():
    """播放闪电音效"""
    if lit_ok and lit_snd is not None and not d_trig and not b3_ok:
        try:
            lit_snd.set_volume(1.0)
            lit_snd.play()
        except:
            pass


def play_rand():
    """随机播放攻击音效"""
    if a_ok and a_snds and not d_trig and not b3_ok:
        vs = [s for s in a_snds if s is not None]
        if vs:
            random.choice(vs).play()


def play_dth():
    """播放死亡音效"""
    global d_trig
    if d_ok and d_snd is not None:
        stop_all()
        d_snd.set_volume(1.0)
        d_snd.play()


def play_atk():
    """播放攻击音效"""
    if atk_ok and atk_snd is not None and not d_trig and not b3_ok:
        atk_snd.set_volume(1.0)
        atk_snd.play()


def play_dan():
    """播放危险提示音"""
    if dan_ok and dan_snd is not None and not d_trig and not b3_ok:
        dan_snd.set_volume(1.0)
        dan_snd.play()


def play_exc():
    """播放大招音效"""
    if exc_ok and exc_snd is not None:
        try:
            exc_snd.set_volume(1.0)
            exc_snd.play()
        except:
            pass


def play_man():
    """播放终局音效"""
    if m_ok and m_snd is not None and not d_trig:
        try:
            stop_all()
            m_snd.set_volume(1.0)
            m_snd.play()
        except:
            pass


def overlay(bg, fg, m, x, y, tr=1.0):
    """
    叠加带透明度的前景图到背景
    bg: 背景图
    fg: 前景图
    m: 遮罩
    x,y: 位置
    tr: 透明度
    """
    h, w = fg.shape[:2]
    xe, ye = x + w, y + h
    cl, ct, cr, cb = 0, 0, w, h
    # 边界裁剪
    if x < 0: cl, x = -x, 0
    if y < 0: ct, y = -y, 0
    if xe > bw: cr, xe = w - (xe - bw), bw
    if ye > bh: cb, ye = h - (ye - bh), ye
    if ct >= 0 and cb <= h and cl >= 0 and cr <= w:
        fg_c = fg[ct:cb, cl:cr]
        m_c = m[ct:cb, cl:cr]
    else:
        fg_c, m_c = fg, m
    if fg_c.size == 0 or ye <= y or xe <= x:
        return bg
    bg_roi = bg[y:ye, x:xe]
    if fg_c.shape[:2] != bg_roi.shape[:2]:
        fg_c = cv2.resize(fg_c, (bg_roi.shape[1], bg_roi.shape[0]))
        m_c = cv2.resize(m_c, (bg_roi.shape[1], bg_roi.shape[0]))
    if len(m_c.shape) > 2:
        m_c = cv2.cvtColor(m_c, cv2.COLOR_BGR2GRAY)
    m_n = cv2.cvtColor(m_c, cv2.COLOR_GRAY2BGR).astype(float) / 255.0 * tr
    if len(fg_c.shape) == 2:
        fg_c = cv2.cvtColor(fg_c, cv2.COLOR_GRAY2BGR)
    res = bg.copy()
    res[y:ye, x:xe] = (fg_c * m_n + bg_roi * (1 - m_n)).astype(np.uint8)
    return res


def chk_cross(lm, lx, w, h):
    """
    检测指尖是否穿过竖直线
    lm: 手部关键点
    lx: 线X坐标
    w,h: 图像宽高
    """
    tx = int(lm.landmark[8].x * w)
    if tx < lx - l_th:
        return -1  # 在线左侧
    elif tx > lx + l_th:
        return 1  # 在线右侧
    return 0  # 未穿过


def chk_b3_cross(lm, v, o, w, h, th=b3_th):
    """
    检测指尖是否接近终局线
    lm: 手部关键点
    v: 线位置
    o: 方向('vertical'/'horizontal')
    th: 判定阈值
    """
    if o == 'vertical':
        tx = int(lm.landmark[8].x * w)
        return abs(tx - v) <= th
    elif o == 'horizontal':
        ty = int(lm.landmark[8].y * h)
        return abs(ty - v) <= th
    return False


def nxt_bgi():
    """获取下一个背景索引"""
    global seq_i, is_init
    if is_init:
        is_init = False
        seq_i = 0
    else:
        seq_i = (seq_i + 1) % len(b_seq) if b_lp else min(seq_i + 1, len(b_seq) - 1)
    return b_seq[seq_i]


def nxt_bgi_w():
    """预览下一个背景索引(不更新索引)"""
    if is_init:
        return b_seq[0]
    elif seq_i < len(b_seq) - 1:
        return b_seq[seq_i + 1]
    return b_seq[0] if b_lp else b_seq[seq_i]


# ===================== 人脸替换函数 =====================
def swap_face_mesh(bg, frame):
    """
    将摄像头人脸融合到背景图的指定位置
    bg: 背景图 (excution.png)
    frame: 摄像头帧
    """
    # MediaPipe FaceMesh人脸检测：将BGR帧转为RGB后处理，获取468个人脸关键点
    fmr = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rh, rw = frame.shape[:2]
    out = bg.copy()

    # 如果检测到人脸，进行融合处理
    if fmr and fmr.multi_face_landmarks:
        # 提取所有关键点坐标并计算边界框
        lms = fmr.multi_face_landmarks[0].landmark
        pts = [(int(p.x * rw), int(p.y * rh)) for p in lms]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(xs) - min(ys)

        if w > 0 and h > 0:
            # 裁剪摄像头人脸区域
            xe = min(rw, x + w)
            ye = min(rh, y + h)
            src = frame[y:ye, x:xe]

            # 计算目标位置：根据预设比例FACE_RATIO在背景图上定位人脸区域
            bh0, bw0 = out.shape[:2]
            fx = int(FACE_RATIO[0] * bw0)
            fy = int(FACE_RATIO[1] * bh0)
            fw = int(FACE_RATIO[2] * bw0)
            fh = int(FACE_RATIO[3] * bh0)

            if fw > 0 and fh > 0 and src.size > 0:
                # 提取面部轮廓点：使用FACEMESH_FACE_OVAL获取面部椭圆边缘点
                oval_idx = set()
                for a, b in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
                    oval_idx.add(a)
                    oval_idx.add(b)

                # 将轮廓点坐标转换到目标区域坐标系
                rel = []
                for i in oval_idx:
                    px = int((pts[i][0] - x) * (fw / max(1, w)))
                    py = int((pts[i][1] - y) * (fh / max(1, h)))
                    rel.append([px, py])
                rel = np.array(rel, dtype=np.int32)

                if rel.shape[0] >= 3:
                    # 计算凸包：获取面部轮廓的凸包用于创建精确遮罩
                    hull = cv2.convexHull(rel)
                    src_r = cv2.resize(src, (fw, fh))
                    dst_roi = out[fy:fy + fh, fx:fx + fw]

                    # LAB色彩空间匹配：将源图像与目标区域进行颜色和亮度匹配
                    src_lab = cv2.cvtColor(src_r, cv2.COLOR_BGR2LAB)
                    dst_lab = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2LAB)
                    for i in range(3):
                        s_m = float(src_lab[..., i].mean())
                        s_s = float(src_lab[..., i].std()) + 1e-6
                        d_m = float(dst_lab[..., i].mean())
                        d_s = float(dst_lab[..., i].std()) + 1e-6
                        # 应用色彩转移公式：(源-源均值)*(目标标准差/源标准差)+目标均值
                        src_lab[..., i] = np.clip((src_lab[..., i] - s_m) * (d_s / s_s) + d_m, 0, 255)
                    src_corr = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

                    # 创建羽化遮罩：使用高斯模糊实现边缘平滑过渡
                    m = np.zeros((fh, fw), dtype=np.uint8)
                    cv2.fillConvexPoly(m, hull, 255)
                    br = max(7, (min(fw, fh) // 10) | 1)  # 计算模糊半径（奇数）
                    m = cv2.GaussianBlur(m, (br, br), 0)

                    # 预处理目标区域：先对背景区域进行高斯模糊
                    base = out.copy()
                    base[fy:fy + fh, fx:fx + fw] = cv2.GaussianBlur(dst_roi, (br, br), 0)

                    # 无缝克隆：使用OpenCV泊松融合实现自然融合效果
                    c = (fx + fw // 2, fy + fh // 2)
                    try:
                        out = cv2.seamlessClone(src_corr, base, m, c, cv2.NORMAL_CLONE)
                    except Exception:
                        # 异常处理回退：如果seamlessClone失败，使用alpha混合
                        alpha = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                        blend = (src_corr.astype(float) * alpha + dst_roi.astype(float) * (1 - alpha)).astype(np.uint8)
                        out[fy:fy + fh, fx:fx + fw] = blend

    return out


# ===================== 手势交互处理函数 =====================
def handle_tutorial_gesture(res, dw, dh, draw, t, prev_ix, current_t, last_swipe_t, current_page, tt_ok, tt_snd):
    """
    处理教程界面的手势交互
    返回: (should_exit, exit_value, new_prev_ix, new_current_t, new_last_swipe_t, swipe_detected, swipe_dir)
    """
    should_exit = False
    exit_value = None
    swipe_detected = False
    swipe_dir = None

    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            ix, iy = int(lm.landmark[8].x * dw), int(lm.landmark[8].y * dh)
            tx, ty = int(lm.landmark[4].x * dw), int(lm.landmark[4].y * dh)

            cv2.circle(draw, (ix, iy), 6, (0, 255, 0), -1)
            cv2.circle(draw, (tx, ty), 6, (0, 0, 255), -1)

            dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
            # 检测是否在返回按钮区域
            in_back = (TUTORIAL_BACK_BTN[0] <= ix <= TUTORIAL_BACK_BTN[0] + TUTORIAL_BACK_BTN[2] and
                       TUTORIAL_BACK_BTN[1] <= iy <= TUTORIAL_BACK_BTN[1] + TUTORIAL_BACK_BTN[3])

            # 检测是否在退出按钮区域
            in_exit = False
            if current_page == 2:
                in_exit = (TUTORIAL_EXIT_BTN[0] <= ix <= TUTORIAL_EXIT_BTN[0] + TUTORIAL_EXIT_BTN[2] and
                           TUTORIAL_EXIT_BTN[1] <= iy <= TUTORIAL_EXIT_BTN[1] + TUTORIAL_EXIT_BTN[3])

            # 捏合手势检测
            if dist < PINCH_TH:
                if in_back:
                    if tt_ok and tt_snd is not None:
                        tt_snd.stop()
                    should_exit = True
                    exit_value = None
                if in_exit:
                    if tt_ok and tt_snd is not None:
                        tt_snd.stop()
                    should_exit = True
                    exit_value = None

            # 滑动检测
            if prev_ix is not None and (t - current_t) > 0.05 and (t - last_swipe_t) > TUTORIAL_COOLDOWN:
                delta_x = ix - prev_ix
                if abs(delta_x) > TUTORIAL_SWIPE_THRESH:
                    swipe_detected = True
                    swipe_dir = "left" if delta_x < 0 else "right"
                    current_t = t
                    last_swipe_t = t

            prev_ix = ix

    return should_exit, exit_value, prev_ix, current_t, last_swipe_t, swipe_detected, swipe_dir


def handle_start_gesture(res, dw, dh, draw, btn_start, btn_tut, strt_ok, strt_snd):
    """
    处理开始界面的手势交互
    返回: (action, value) - action可以是"start", "tutorial", "none"
    """
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            ix, iy = int(lm.landmark[8].x * dw), int(lm.landmark[8].y * dh)
            tx, ty = int(lm.landmark[4].x * dw), int(lm.landmark[4].y * dh)
            cv2.circle(draw, (ix, iy), 6, (0, 255, 0), -1)
            cv2.circle(draw, (tx, ty), 6, (0, 0, 255), -1)
            dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

            # 检测按钮点击
            in_start = (btn_start[0] <= ix <= btn_start[0] + btn_start[2] and
                        btn_start[1] <= iy <= btn_start[1] + btn_start[3])
            in_tut = (btn_tut[0] <= ix <= btn_tut[0] + btn_tut[2] and
                      btn_tut[1] <= iy <= btn_tut[1] + btn_tut[3])

            if dist < PINCH_TH:
                if in_start:
                    if strt_ok and strt_snd is not None:
                        strt_snd.stop()
                    return "start", True
                elif in_tut:
                    if strt_ok and strt_snd is not None:
                        strt_snd.stop()
                    return "tutorial", None

    return "none", None


def handle_end_gesture(res, dw, dh, draw, btn_quit, btn_restart):
    """
    处理结束界面的手势交互
    返回: (should_exit, restart) - should_exit是否退出程序, restart是否重新开始游戏
    """
    should_exit = False
    restart = False

    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            ix, iy = int(lm.landmark[8].x * dw), int(lm.landmark[8].y * dh)
            tx, ty = int(lm.landmark[4].x * dw), int(lm.landmark[4].y * dh)
            cv2.circle(draw, (ix, iy), 6, (0, 255, 0), -1)
            cv2.circle(draw, (tx, ty), 6, (0, 0, 255), -1)
            dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

            in_quit = (btn_quit[0] <= ix <= btn_quit[0] + btn_quit[2] and
                       btn_quit[1] <= iy <= btn_quit[1] + btn_quit[3])
            in_restart = (btn_restart[0] <= ix <= btn_restart[0] + btn_restart[2] and
                          btn_restart[1] <= iy <= btn_restart[1] + btn_restart[3])

            if dist < PINCH_TH:
                if in_quit:
                    return True, False
                if in_restart:
                    return True, True

    return False, False


# ===================== 主程序入口 =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

cv2.namedWindow('Background with Lines', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Background with Lines', int(bw * ds), int(bh * ds))


def tutorial_screen():
    """教程界面"""
    tt_path = os.path.join(A_F, a_tt)
    tt_snd = pygame.mixer.Sound(tt_path) if os.path.exists(tt_path) else None
    tt_ok = tt_snd is not None
    if tt_ok and tt_snd is not None:
        tt_snd.set_volume(1.0)
        tt_snd.play(-1)  # 循环播放

    # 加载教程图片
    tutorial_imgs = []
    texts = [
        "Touch the red line to defend",
        "Touch three lines quickly",
        "Grasp!"
    ]
    text_positions = [(50, 200), (60, 200), (150, 200)]
    font_scales = [0.7, 0.7, 1.0]

    for i in range(1, 4):
        path = os.path.join(TUTORIAL_FOLDER, f"{i}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (dw, dh))
            cv2.putText(img, texts[i - 1], text_positions[i - 1],
                        cv2.FONT_HERSHEY_SIMPLEX, font_scales[i - 1], (0, 0, 255), 2)
            if i == 3:
                # 绘制退出按钮
                cv2.rectangle(img, (TUTORIAL_EXIT_BTN[0], TUTORIAL_EXIT_BTN[1]),
                              (TUTORIAL_EXIT_BTN[0] + TUTORIAL_EXIT_BTN[2],
                               TUTORIAL_EXIT_BTN[1] + TUTORIAL_EXIT_BTN[3]),
                              (0, 0, 255), -1)
                cv2.putText(img, "EXIT", (TUTORIAL_EXIT_BTN[0] + 20, TUTORIAL_EXIT_BTN[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tutorial_imgs.append(img)
        else:
            # 占位图
            placeholder = np.zeros((dh, dw, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Tutorial {i}", (150, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            tutorial_imgs.append(placeholder)

    current_page = 0
    prev_ix = None
    current_t = 0
    last_swipe_t = 0

    # 教程主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            if tt_ok and tt_snd is not None:
                tt_snd.stop()
            return

        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t = time.time()

        draw = tutorial_imgs[current_page].copy()

        # 绘制返回按钮
        cv2.rectangle(draw, (TUTORIAL_BACK_BTN[0], TUTORIAL_BACK_BTN[1]),
                      (TUTORIAL_BACK_BTN[0] + TUTORIAL_BACK_BTN[2],
                       TUTORIAL_BACK_BTN[1] + TUTORIAL_BACK_BTN[3]),
                      (200, 200, 200), -1)
        cv2.putText(draw, "BACK", (TUTORIAL_BACK_BTN[0] + 10, TUTORIAL_BACK_BTN[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 显示摄像头画面
        out_cam = cv2.resize(frame, (cdw, cdh))
        draw[0:cdh, 0:cdw] = out_cam

        swipe_detected = False
        swipe_dir = None

        # 调用教程手势处理函数
        result = handle_tutorial_gesture(res, dw, dh, draw, t, prev_ix, current_t, last_swipe_t, current_page, tt_ok,
                                         tt_snd)
        if result[0]:
            return result[1]
        prev_ix = result[2]
        current_t = result[3]
        last_swipe_t = result[4]
        swipe_detected = result[5]
        swipe_dir = result[6]

        # 处理滑动翻页
        if swipe_detected:
            if swipe_dir == "left" and current_page < 2:
                current_page += 1
                prev_ix = None
            elif swipe_dir == "right" and current_page > 0:
                current_page -= 1
                prev_ix = None

        cv2.imshow('Background with Lines', cv2.resize(draw, (int(bw * ds), int(bh * ds))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if tt_ok and tt_snd is not None:
                tt_snd.stop()
            return


def start_screen():
    """开始界面"""
    btn_start = (80, 250, 100, 60)  # 开始按钮区域
    btn_tut = (220, 250, 100, 60)  # 教程按钮区域

    # 播放背景音乐
    if strt_ok and strt_snd is not None:
        strt_snd.stop()
        strt_snd.set_volume(1.0)
        strt_snd.play(-1)

    # 开始界面循环
    while True:
        ret, frame = cap.read()
        if not ret:
            if strt_ok and strt_snd is not None:
                strt_snd.stop()
            return False

        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = start_bg.copy() if start_bg_ok else np.zeros((dh, dw, 3), np.uint8)

        # 绘制按钮
        cv2.rectangle(draw, (btn_start[0], btn_start[1]),
                      (btn_start[0] + btn_start[2], btn_start[1] + btn_start[3]),
                      (0, 255, 0), -1)
        cv2.putText(draw, "START", (btn_start[0] + 18, btn_start[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(draw, (btn_tut[0], btn_tut[1]),
                      (btn_tut[0] + btn_tut[2], btn_tut[1] + btn_tut[3]),
                      (200, 200, 200), -1)
        cv2.putText(draw, "TUTORIAL", (btn_tut[0], btn_tut[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 显示摄像头
        out_cam = cv2.resize(frame, (cdw, cdh))
        draw[0:cdh, 0:cdw] = out_cam

        # 调用开始界面手势处理函数
        action, value = handle_start_gesture(res, dw, dh, draw, btn_start, btn_tut, strt_ok, strt_snd)
        if action == "start":
            return value
        elif action == "tutorial":
            if strt_ok and strt_snd is not None:
                strt_snd.stop()
            tutorial_screen()
            if strt_ok and strt_snd is not None:
                strt_snd.stop()
                strt_snd.set_volume(1.0)
                strt_snd.play(-1)

        cv2.imshow('Background with Lines', cv2.resize(draw, (int(bw * ds), int(bh * ds))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if strt_ok and strt_snd is not None:
                strt_snd.stop()
            return False


def end_screen():
    """结束界面"""
    btn_quit = (80, 250, 100, 60)  # 退出按钮
    btn_restart = (220, 250, 100, 60)  # 重新开始按钮
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = t_img.copy() if t_ok else np.zeros((dh, dw, 3), np.uint8)

        # 绘制按钮
        cv2.rectangle(draw, (btn_quit[0], btn_quit[1]),
                      (btn_quit[0] + btn_quit[2], btn_quit[1] + btn_quit[3]),
                      (0, 0, 255), -1)
        cv2.putText(draw, "EXIT", (btn_quit[0] + 25, btn_quit[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(draw, (btn_restart[0], btn_restart[1]),
                      (btn_restart[0] + btn_restart[2], btn_restart[1] + btn_restart[3]),
                      (0, 255, 0), -1)
        cv2.putText(draw, "RESTART", (btn_restart[0] + 5, btn_restart[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 调用结束界面手势处理函数
        should_exit, restart = handle_end_gesture(res, dw, dh, draw, btn_quit, btn_restart)
        if should_exit:
            return restart

        cv2.imshow('Background with Lines', cv2.resize(draw, (int(bw * ds), int(bh * ds))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False


# ===================== 进入主循环 =====================
if not start_screen():
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pygame.mixer.quit()
    exit()

# ===================== 游戏状态变量初始化 =====================
# 闪电相关
llx = int(bw * l_lr)  # 左线X坐标
lrx = int(bw * l_rr)  # 右线X坐标
lit_t0 = 0  # 闪电开始时间
lit_on = False  # 闪电是否激活
main_paths = []  # 主闪电路径
all_br = []  # 分支路径

# 终局阶段相关
b3_tc = False  # 顶线是否完成
b3_lc = False  # 左线是否完成
b3_rc = False  # 右线是否完成
b3_ta = False  # 顶线是否激活
b3_la = False  # 左线是否激活
b3_ra = False  # 右线是否激活
b3_all = False  # 是否全部完成
b3_fail = False  # 是否失败
b3_ok = False  # 终局是否成功
b3_t0 = 0  # 终局开始时间
b3_dur = 3.0  # 终局持续时间

# 倒计时相关
s_on = False  # 倒计时是否激活
s_t0 = 0  # 倒计时开始时间

# 抓取检测相关
g_det = False  # 是否检测到抓取
g_act = False  # 是否激活抓取
g_t0 = 0  # 抓取检测时间

# 大招相关
e_on = False  # 大招是否激活
e_t0 = 0  # 大招开始时间
e_dur = 3.0  # 大招持续时间

# 游戏进度相关
prog_t0 = time.time()  # 进度开始时间
cbi = 0  # 当前背景索引
b_chg_t0 = time.time()  # 背景切换时间
seq_i = 0  # 序列索引
is_init = True  # 是否初始状态
w_trig = False  # 武器触发
l_trig = False  # 左触发
r_trig = False  # 右触发
chp = 3  # 当前HP
l_lt0 = 0  # 左触发时间
r_lt0 = 0  # 右触发时间
dan_blk = {1: False, 2: False}  # 危险状态记录

# 死亡相关
d_trig = False  # 死亡触发
d_t0 = 0  # 死亡开始时间
d_aud = False  # 死亡音效是否播放

# 姿势相关
cp = 0  # 姿势索引
pst_t0 = 0  # 姿势显示时间
pst_show = False  # 是否显示姿势
pst_shown = False  # 是否显示过姿势
pb_i = 0  # 前背景索引
fst_dan = False  # 首次危险
pst_max = False  # 姿势是否最大
show_dan = False  # 是否显示危险提示
nxt_bi = 0  # 下一个背景索引
b3_z = False  # 终局预警
dan_aud = False  # 危险音效是否播放
ext_on = False  # 终局扩展
ext_t0 = 0  # 扩展开始时间
b3_ext = False  # 终局激活
b3_et0 = 0  # 终局激活时间
dan_t0 = 0  # 危险提示开始时间
dan_fp = 0.0  # 危险提示透明度
r_delay_t0 = 0  # 红屏延迟时间

# ===================== 主游戏循环 =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cam_frm = cv2.resize(frame, (cdw, cdh))
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 重置每帧状态
    l_trig = r_trig = k_vis = k_flip = False
    kx = ky = 0
    b3_tt = b3_lt = b3_rt = False
    t = time.time()
    g_cur = False

    # 手势处理
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            tx = int(lm.landmark[8].x * bw)  # 食指X
            ty = int(lm.landmark[8].y * bh)  # 食指Y

            # 武器位置跟随
            if not e_on and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on:
                if k_ok:
                    kx = tx - (k_img.shape[1] // 2)
                    ky = ty - (k_img.shape[0] // 2)
                    k_vis = True
                    k_flip = tx > bw // 2  # 根据位置翻转武器

            # 检测线穿越
            if not e_on and cbi != 3 and not d_trig and not b3_ok and not s_on:
                ls = chk_cross(lm, llx, bw, bh)
                if ls == -1 and not l_trig:
                    l_trig = True
                elif ls != -1:
                    l_trig = False

                rs = chk_cross(lm, lrx, bw, bh)
                if rs == 1 and not r_trig:
                    r_trig = True
                elif rs != 1:
                    r_trig = False

            # 检测终局线接近
            if not e_on and cbi == 3 and b3_ext and not d_trig and not b3_all and not b3_fail and not b3_ok and not s_on:
                if not b3_tc and chk_b3_cross(lm, b3_ty, 'horizontal', bw, bh):
                    if not b3_ta:
                        b3_tt = b3_ta = True
                        b3_tc = True
                        threading.Thread(target=play_rand, daemon=True).start()
                elif not chk_b3_cross(lm, b3_ty, 'horizontal', bw, bh):
                    b3_ta = False

                if not b3_lc and chk_b3_cross(lm, b3_lx, 'vertical', bw, bh):
                    if not b3_la:
                        b3_lt = b3_la = True
                        b3_lc = True
                        threading.Thread(target=play_rand, daemon=True).start()
                elif not chk_b3_cross(lm, b3_lx, 'vertical', bw, bh):
                    b3_la = False

                if not b3_rc and chk_b3_cross(lm, b3_rx, 'vertical', bw, bh):
                    if not b3_ra:
                        b3_rt = b3_ra = True
                        b3_rc = True
                        threading.Thread(target=play_rand, daemon=True).start()
                elif not chk_b3_cross(lm, b3_rx, 'vertical', bw, bh):
                    b3_ra = False

            # 检测抓取手势(大招)
            if pst_max and r_ok and not e_on and not d_trig and not b3_fail and not b3_ok and not s_on:
                if t - g_t0 >= g_iv:
                    g_cur = chk_grasp(lm, bw, bh)
                    if g_cur and not g_act:
                        g_det = True
                        g_act = True
                        e_on = True
                        e_t0 = t
                        stop_all()
                        if exc_ok:
                            try:
                                exc_snd.set_volume(1.0)
                                exc_snd.play()
                            except:
                                pass
                    elif not g_cur:
                        g_act = False
                    g_t0 = t
    else:
        g_act = False

    elp = t - prog_t0  # 经过时间

    # 终局阶段逻辑
    if not e_on and cbi == 3 and b3_ext and not d_trig and not b3_all and not b3_fail and not b3_ok and not s_on:
        if b3_tc and b3_lc and b3_rc:
            b3_all = True
            threading.Thread(target=play_rand, daemon=True).start()
            b3_ok = True
            b3_t0 = t
            stop_all()
            play_man()  # 播放终局音效
            cbi = 4
            s_on = True
            s_t0 = t

    # 倒计时结束
    if s_on and t - s_t0 >= s_dt:
        s_on = False
        b3_ok = False
        b3_ext = False
        b3_tc = False
        b3_lc = False
        b3_rc = False
        b3_all = False
        b3_fail = False
        pb_i = 0
        cbi = 0
        b_chg_t0 = t
        is_init = False
        seq_i = 0
        ext_on = False
        b3_z = False
        dan_aud = False

    # 终局超时失败
    if b3_ok and not s_on and t - b3_t0 >= b3_dur:
        b3_ok = False
        b3_ext = False
        b3_tc = False
        b3_lc = False
        b3_rc = False
        b3_all = False
        b3_fail = False
        pb_i = 0
        cbi = 0
        b_chg_t0 = t
        is_init = False
        seq_i = 0
        ext_on = False
        b3_z = False
        dan_aud = False

    # 大招结束
    if e_on:
        if e_bg_ok:
            out = swap_face_mesh(e_bg, frame)

        # 大招结束
        if t - e_t0 >= e_dur:
            if end_ok and end_snd is not None:
                try:
                    end_snd.set_volume(1.0)
                    end_snd.play()
                except:
                    pass

            if t_ok:
                out = t_img.copy()
            else:
                out = bgs[0].copy()

            cv2.imshow('Background with Lines', cv2.resize(out, (int(bw * ds), int(bh * ds))))
            cv2.waitKey(1000)

            restart = end_screen()

            if end_ok and end_snd is not None:
                try:
                    end_snd.stop()
                except:
                    pass

            if restart:
                # 重置所有状态
                pygame.mixer.stop()
                prog_t0 = time.time()
                cbi = 0
                b_chg_t0 = time.time()
                seq_i = 0
                is_init = True
                w_trig = False
                l_trig = False
                r_trig = False
                chp = 3
                l_lt0 = 0
                r_lt0 = 0
                dan_blk = {1: False, 2: False}
                d_trig = False
                d_t0 = 0
                d_aud = False
                cp = 0
                pst_t0 = 0
                pst_show = False
                pst_shown = False
                pb_i = 0
                fst_dan = False
                pst_max = False
                show_dan = False
                nxt_bi = 0
                b3_z = False
                dan_aud = False
                ext_on = False
                ext_t0 = 0
                b3_ext = False
                b3_et0 = 0
                dan_t0 = 0
                dan_fp = 0.0

                lit_t0 = 0
                lit_on = False
                main_paths = []
                all_br = []

                b3_tc = False
                b3_lc = False
                b3_rc = False
                b3_ta = False
                b3_la = False
                b3_ra = False
                b3_all = False
                b3_fail = False
                b3_ok = False
                b3_t0 = 0
                b3_dur = 3.0
                s_on = False
                s_t0 = 0

                g_det = False
                g_act = False
                g_t0 = 0
                e_on = False
                e_t0 = 0
                e_dur = 3.0
                continue
            else:
                break

    # 死亡或失败状态
    elif d_trig or pst_max or b3_fail:
        cbi = 0
    else:
        # 背景切换逻辑
        if elp < b_it:
            cbi = 0
        else:
            if is_init:
                cbi = b_seq[0]
                is_init = False
                b_chg_t0 = t
                pb_i = 0
                if cbi in [1, 2]:
                    threading.Thread(target=play_atk, daemon=True).start()
                    fst_dan = True
            elif not b3_ok and not s_on and t - b_chg_t0 > b_iv:
                if b3_ext:
                    # 终局超时检测
                    if t - b3_et0 >= b3_dt:
                        if not b3_all and not b3_fail and not b3_ok and not s_on:
                            b3_fail = True
                            d_trig = True
                            d_t0 = t
                            play_dth()
                            d_aud = True
                elif ext_on:
                    # 进入终局
                    if t - ext_t0 >= b3_et:
                        ext_on = False
                        b3_z = False
                        dan_aud = False
                        pb_i = cbi
                        cbi = 3
                        b_chg_t0 = t
                        b3_ext = True
                        b3_et0 = t
                        b3_tc = False
                        b3_lc = False
                        b3_rc = False
                        b3_all = False
                        b3_fail = False
                        threading.Thread(target=play_atk, daemon=True).start()
                elif not b3_ok and not s_on:
                    # 正常背景切换
                    pb_i = cbi
                    nxt = nxt_bgi()
                    if cbi == 0 and nxt == 3:
                        b3_z = True
                        ext_on = True
                        ext_t0 = t
                        if dan_ok and not dan_aud:
                            threading.Thread(target=play_dan, daemon=True).start()
                            dan_aud = True
                    else:
                        # HP扣减逻辑
                        if pb_i in [1, 2] and not dan_blk[pb_i]:
                            chp = max(0, chp - 1)
                            if chp == 0 and not d_trig:
                                d_trig = True
                                d_t0 = t
                                play_dth()
                                d_aud = True
                        dan_blk[1] = False
                        dan_blk[2] = False
                        cbi = nxt
                        b_chg_t0 = t
                        if cbi in [1, 2] and pb_i != cbi:
                            threading.Thread(target=play_atk, daemon=True).start()

    # 危险提示动画
    show_dan0 = show_dan
    show_dan = (not d_trig and not pst_max and not b3_fail and not e_on and not b3_ok and not s_on and
                ((cbi == 0 and nxt_bgi_w() == 3) or ext_on))
    if show_dan:
        if not show_dan0:
            dan_t0 = t
            dan_fp = 0.0
        else:
            ft = t - dan_t0
            if ft < dan_fd:
                dan_fp = min(1.0, ft / dan_fd)  # 淡入
            else:
                dan_fp = 1.0
    else:
        dan_fp = 0.0
        dan_t0 = 0

    # 绘制背景
    if not e_on and (cbi == 3 or b3_ok) and not d_trig and not pst_max and not b3_fail and not s_on:
        if b3_ok:
            out = bgs[4].copy()
        else:
            out = bgs[3].copy()
        # 生成闪电
        if not lit_on and t - lit_t0 > l_iv:
            lit_on = True
            lit_t0 = t
            main_paths = []
            all_br = []
            if lit_ok and not b3_ok:
                threading.Thread(target=play_lit, daemon=True).start()
            nl = random.randint(l_cn, l_cx)
            for _ in range(nl):
                sx = random.randint(0, bw)
                ex = random.randint(0, bw)
                sp, ep = (sx, 0), (ex, bh)
                ns = random.randint(l_ms - 5, l_ms)
                jg = random.randint(l_jn, l_jx)
                path = gen_lit_path(sp, ep, ns=ns, j=jg)
                main_paths.append(path)
                if path and random.random() > 0.3:
                    nb = random.randint(1, l_mb)
                    for _ in range(nb):
                        si = random.randint(1, len(path) - 2)
                        bs = path[si]
                        be = (bs[0] + random.randint(-100, 100), bs[1] + random.randint(-80, 80))
                        branch = gen_lit_path(bs, be, ns=random.randint(4, 8),
                                              j=random.randint(l_jn // 2, l_jx // 2))
                        all_br.append(branch)
        if lit_on:
            out = draw_lit(out, main_paths, all_br)
            if t - lit_t0 > l_du:
                lit_on = False
    elif not e_on and not s_on:
        out = bgs[cbi].copy()
        lit_on = False
        main_paths = []
        all_br = []
    elif s_on:
        out = bgs[4].copy()

        # 倒计时文字
        show_text = (t < s_t0 + s_ld)
        if show_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.5
            thickness = 3

            # 第一行：Lightning（黄色）
            txt1 = "Lightning"
            (tw1, th1), _ = cv2.getTextSize(txt1, font, scale, thickness)
            tx1 = (bw - tw1) // 2
            ty1 = bh // 2 - 5  # 上半部分位置

            # 第二行：Counterattack（黄色）
            txt2 = "Counterattack"
            (tw2, th2), _ = cv2.getTextSize(txt2, font, scale, thickness)
            tx2 = (bw - tw2) // 2
            ty2 = bh // 2 + th2 + 5  # 下半部分位置

            # 绘制第一行文字（黑色描边 + 黄色填充）
            cv2.putText(out, txt1, (tx1, ty1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, txt1, (tx1, ty1), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

            # 绘制第二行文字（黑色描边 + 黄色填充）
            cv2.putText(out, txt2, (tx2, ty2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, txt2, (tx2, ty2), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        # 倒计时闪电
        if t - s_t0 >= s_ld:
            if not lit_on and t - lit_t0 > l_iv:
                lit_on = True
                lit_t0 = t
                main_paths = []
                all_br = []
                if lit_ok:
                    threading.Thread(target=play_lit, daemon=True).start()
                nl = random.randint(l_cn, l_cx)
                for _ in range(nl):
                    sx = random.randint(0, bw)
                    ex = random.randint(0, bw)
                    sp, ep = (sx, 0), (ex, bh)
                    ns = random.randint(l_ms - 5, l_ms)
                    jg = random.randint(l_jn, l_jx)
                    path = gen_lit_path(sp, ep, ns=ns, j=jg)
                    main_paths.append(path)
                    if path and random.random() > 0.3:
                        nb = random.randint(1, l_mb)
                        for _ in range(nb):
                            si = random.randint(1, len(path) - 2)
                            bs = path[si]
                            be = (bs[0] + random.randint(-100, 100), bs[1] + random.randint(-80, 80))
                            branch = gen_lit_path(bs, be, ns=random.randint(4, 8),
                                                  j=random.randint(l_jn // 2, l_jx // 2))
                            all_br.append(branch)
            if lit_on:
                out = draw_lit(out, main_paths, all_br)
                if t - lit_t0 > l_du:
                    lit_on = False
        else:
            lit_on = False
            main_paths = []
            all_br = []

    # 绘制终局线
    if not e_on and cbi == 3 and b3_ext and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on:
        lc = b3_cc if b3_tc else b3_ic
        cv2.line(out, (0, b3_ty), (bw, b3_ty), lc, b3_w)
        lc = b3_cc if b3_lc else b3_ic
        cv2.line(out, (b3_lx, 0), (b3_lx, bh), lc, b3_w)
        lc = b3_cc if b3_rc else b3_ic
        cv2.line(out, (b3_rx, 0), (b3_rx, bh), lc, b3_w)

    # 叠加武器
    if not e_on and not pst_max and not d_trig and not b3_fail and not b3_ok and not s_on:
        if k_ok and k_vis:
            if k_flip:
                out = overlay(out, k_img_f, k_msk_f, kx, ky)
            else:
                out = overlay(out, k_img, k_msk, kx, ky)

    # 叠加HP条
    if not e_on and not d_trig and h_ok and 0 <= chp < len(h_imgs) and not s_on:
        if h_pos == "bottom_left":
            hx, hy = h_xo, bh - hh - h_yo
        out = overlay(out, h_imgs[chp], h_msks[chp], hx, hy)

    # 叠加死亡画面
    if not e_on and d_trig and d_ok:
        dx = (bw - ndw) // 2
        dy = (bh - ndh) // 2
        out = overlay(out, d_img, d_msk, dx, dy)
        if t - d_t0 >= d_dt:
            e_on = True
            e_t0 = t - e_dur

    # 叠加姿势图
    if not e_on and not d_trig and p_ok and pst_shown and not pst_max and not s_on:
        if 0 <= cp < len(p_imgs):
            if p_pos == "top_center":
                px = (bw - pw) // 2 + p_xo
                py = p_yo
            elif p_pos == "top_left":
                px = p_xo
                py = p_yo
            elif p_pos == "top_right":
                px = bw - pw - p_xo
                py = p_yo
            else:
                px = (bw - pw) // 2 + p_xo
                py = p_yo
            out = overlay(out, p_imgs[cp], p_msks[cp], px, py)

    # 叠加危险提示
    if not e_on and show_dan and dan_ok and dan_fp > 0:
        cs = dan_ss + (dan_se - dan_ss) * dan_fp  # 缩放动画
        ct = dan_fp  # 透明度
        ndw = int(bw * dan_sf * cs)
        ndh = int(dan_h0 * ndw / dan_w0)
        dan_img_r = cv2.resize(dan_img, (ndw, ndh))
        dan_msk_r = cv2.resize(dan_msk, (ndw, ndh))
        dx = (bw - ndw) // 2
        dy = (bh - ndh) // 2
        out = overlay(out, dan_img_r, dan_msk_r, dx, dy, ct)

    # 叠加红屏
    if not e_on and not d_trig and pst_max and r_ok and not b3_ok and not s_on:
        rx = (bw - nrw) // 2
        ry = (bh - nrh) // 2
        out = overlay(out, r_img, r_msk, rx, ry, r_tr)

    # 叠加摄像头画面
    if not e_on:
        out[0:cdh, 0:cdw] = cam_frm

    # 绘制预警线
    if not e_on and cbi != 3 and not b3_ok and not s_on and not d_trig and not pst_max:
        cv2.line(out, (llx, 0), (llx, bh), (0, 0, 255), 1)
        cv2.line(out, (lrx, 0), (lrx, bh), (0, 0, 255), 1)

    # 绘制预警覆盖
    if not e_on and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on:
        ov = out.copy()
        nxt = nxt_bgi_w()
        ts = elp
        ti = b_it - ts
        tc = t - b_chg_t0
        tn = b_iv - tc
        in_w = False
        if is_init and ti <= w_at and ti > 0:
            in_w = True
        elif not is_init and tn <= w_at and tn > 0:
            in_w = True
        if in_w:
            if nxt == 1:
                cv2.rectangle(ov, (lrx, 0), (bw, bh), (0, 0, 255), -1)
            elif nxt == 2:
                cv2.rectangle(ov, (0, 0), (llx, bh), (0, 0, 255), -1)
        if cbi == 1:
            cv2.rectangle(ov, (lrx, 0), (bw, bh), (0, 0, 255), -1)
        elif cbi == 2:
            cv2.rectangle(ov, (0, 0), (llx, bh), (0, 0, 255), -1)
        elif cbi == 3 and not b3_ext:
            cv2.rectangle(ov, (0, 0), (bw, bh), (0, 0, 255), -1)
        out = cv2.addWeighted(ov, w_al, out, 1 - w_al, 0)

    # 记录防御状态
    if not e_on and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on and cbi in [1, 2, 3]:
        if cbi == 1 and r_trig:
            dan_blk[1] = True
        elif cbi == 2 and l_trig:
            dan_blk[2] = True

    # 左侧触发事件
    if not e_on and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on and a_ok and (
            t - l_lt0) > a_cd:
        if cbi == 2 and l_trig:
            threading.Thread(target=play_rand, daemon=True).start()
            l_lt0 = t
            if p_ok:
                if not pst_shown:
                    cp = 0
                    pst_shown = True
                else:
                    if cp < len(p_imgs) - 1:
                        cp = (cp + 1) % len(p_imgs)
                        if cp == len(p_imgs) - 1:
                            pst_max = True
                            r_delay_t0 = time.time()

    # 右侧触发事件
    if not e_on and not d_trig and not pst_max and not b3_fail and not b3_ok and not s_on and a_ok and (
            t - r_lt0) > a_cd:
        if cbi == 1 and r_trig:
            threading.Thread(target=play_rand, daemon=True).start()
            r_lt0 = t
            if p_ok:
                if not pst_shown:
                    cp = 0
                    pst_shown = True
                else:
                    if cp < len(p_imgs) - 1:
                        cp = (cp + 1) % len(p_imgs)
                        if cp == len(p_imgs) - 1:
                            pst_max = True
                            r_delay_t0 = time.time()

    # 持续检测抓取
    if pst_max and r_ok and not e_on and not d_trig and not b3_fail and not b3_ok and not s_on:
        if t - g_t0 >= g_iv:
            g_cur = chk_grasp(lm, bw, bh)
            if g_cur and not g_act:
                g_det = True
                g_act = True
                e_on = True
                e_t0 = t
                stop_all()
                if exc_ok:
                    try:
                        exc_snd.set_volume(1.0)
                        exc_snd.play()
                    except:
                        pass
            elif not g_cur:
                g_act = False
            g_t0 = t

    # 显示最终画面
    cv2.imshow('Background with Lines', cv2.resize(out, (int(bw * ds), int(bh * ds))))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== 清理资源 =====================
cap.release()
cv2.destroyAllWindows()
hands.close()
try:
    face_det.close()
except Exception:
    pass
try:
    face_mesh.close()
except Exception:
    pass
pygame.mixer.quit()
import vtk
import os

# 设置文件路径和文件名格式
file_path = '../build/'
file_pattern = 'jet_{d}.vtk'
num_frames = 100  # 假设有100个文件

# 创建渲染器
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.2, 0.3)  # 设置背景颜色

# 创建渲染窗口
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(800, 600)

# 创建交互器
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# 创建一个读取器
reader = vtk.vtkUnstructuredGridReader()

# 创建一个映射器
mapper = vtk.vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())

# 创建一个演员
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 将演员添加到渲染器
renderer.AddActor(actor)

# 创建一个动画场景
animationScene = vtk.vtkAnimationScene()
animationScene.SetLoopMode(vtk.vtkAnimationScene.LOOP)
animationScene.SetMaximumFrameRate(30)  # 设置最大帧率
animationScene.SetEndFrame(num_frames - 1)  # 设置结束帧

# 创建一个动画提示
animationCue = vtk.vtkAnimationCue()
animationCue.SetStartFrame(0)
animationCue.SetEndFrame(num_frames - 1)

# 设置动画提示的回调函数
def update_scene(caller, event):
    frame = caller.GetFrame()
    file_name = os.path.join(file_path, file_pattern.format(frame))
    reader.SetFileName(file_name)
    reader.Update()
    renderWindow.Render()

animationCue.AddObserver(vtk.vtkCommand.AnimateCueEvent, update_scene)

# 将动画提示添加到动画场景
animationScene.AddCue(animationCue)

# 开始动画
renderWindow.Render()
animationScene.Play()

# 启动交互器
renderWindowInteractor.Start()
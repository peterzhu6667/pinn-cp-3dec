"""
从3DEC结果导出PINN训练数据

此脚本用于将3DEC计算结果导出为PINN可用的.mat格式数据文件。

使用方法:
    1. 在3DEC中运行模型并保存结果
    2. 使用3DEC的FISH脚本导出数据为文本文件
    3. 运行此脚本将文本数据转换为.mat格式

输出文件: elastic_hole_3dec.mat
"""

import numpy as np
import scipy.io as sio
import os

def read_3dec_results(base_path):
    """
    读取3DEC导出的结果文件
    
    假设3DEC通过FISH脚本已经导出了以下文本文件:
    - nodes.txt: 节点坐标 (x, y, z)
    - displacement.txt: 位移 (u_x, u_y, u_z)
    - stress.txt: 应力 (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz)
    - boundary_hole.txt: 孔边界节点坐标
    - boundary_far.txt: 远场边界节点坐标
    
    返回:
        dict: 包含所有数据的字典
    """
    
    data = {}
    
    # 读取节点坐标
    if os.path.exists(os.path.join(base_path, 'nodes.txt')):
        nodes = np.loadtxt(os.path.join(base_path, 'nodes.txt'))
        data['x'] = nodes[:, 0]
        data['y'] = nodes[:, 1]
        data['z'] = nodes[:, 2]
        print(f"读取 {len(data['x'])} 个节点坐标")
    
    # 读取位移
    if os.path.exists(os.path.join(base_path, 'displacement.txt')):
        disp = np.loadtxt(os.path.join(base_path, 'displacement.txt'))
        data['u_x'] = disp[:, 0]
        data['u_y'] = disp[:, 1]
        data['u_z'] = disp[:, 2]
        print(f"读取 {len(data['u_x'])} 个位移数据")
    
    # 读取应力
    if os.path.exists(os.path.join(base_path, 'stress.txt')):
        stress = np.loadtxt(os.path.join(base_path, 'stress.txt'))
        data['sigma_xx'] = stress[:, 0]
        data['sigma_yy'] = stress[:, 1]
        data['sigma_zz'] = stress[:, 2]
        data['sigma_xy'] = stress[:, 3]
        data['sigma_xz'] = stress[:, 4]
        data['sigma_yz'] = stress[:, 5]
        print(f"读取 {len(data['sigma_xx'])} 个应力数据")
    
    # 读取孔边界
    if os.path.exists(os.path.join(base_path, 'boundary_hole.txt')):
        boundary_hole = np.loadtxt(os.path.join(base_path, 'boundary_hole.txt'))
        data['hole_boundary_x'] = boundary_hole[:, 0]
        data['hole_boundary_y'] = boundary_hole[:, 1]
        data['hole_boundary_z'] = boundary_hole[:, 2]
        print(f"读取 {len(data['hole_boundary_x'])} 个孔边界点")
    
    # 读取远场边界
    if os.path.exists(os.path.join(base_path, 'boundary_far.txt')):
        boundary_far = np.loadtxt(os.path.join(base_path, 'boundary_far.txt'))
        data['far_field_boundary_x'] = boundary_far[:, 0]
        data['far_field_boundary_y'] = boundary_far[:, 1]
        data['far_field_boundary_z'] = boundary_far[:, 2]
        print(f"读取 {len(data['far_field_boundary_x'])} 个远场边界点")
    
    # 添加材料参数
    data['shear_modulus'] = np.array([2.8e9])
    data['bulk_modulus'] = np.array([3.9e9])
    data['poissons_ratio'] = np.array([0.167])
    data['hole_radius'] = np.array([1.0])
    data['far_field_stress_x'] = np.array([15e6])
    data['far_field_stress_y'] = np.array([30e6])
    
    return data


def export_to_mat(data, output_file):
    """
    将数据导出为.mat格式
    
    参数:
        data: dict, 包含所有数据的字典
        output_file: str, 输出文件路径
    """
    sio.savemat(output_file, data)
    print(f"\n数据已导出到: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")


def create_sample_fish_export_script(output_path):
    """
    创建3DEC的FISH导出脚本示例
    
    此函数生成一个.fis文件，可在3DEC中运行以导出数据
    """
    fish_script = """
;---------------------------------------------------------------------
; 3DEC数据导出脚本 - 用于PINN训练
;---------------------------------------------------------------------
fish define export_for_pinn
    ; 导出节点坐标和位移
    local fp_nodes = file.open('nodes.txt', 1, 0)
    local fp_disp = file.open('displacement.txt', 1, 0)
    
    loop foreach local igp_ block.gp.list
        ; 节点坐标
        file.write(fp_nodes, block.gp.pos.x(igp_))
        file.write(fp_nodes, ' ')
        file.write(fp_nodes, block.gp.pos.y(igp_))
        file.write(fp_nodes, ' ')
        file.write(fp_nodes, block.gp.pos.z(igp_))
        file.write(fp_nodes, 13)  ; 换行
        
        ; 位移
        file.write(fp_disp, block.gp.disp.x(igp_))
        file.write(fp_disp, ' ')
        file.write(fp_disp, block.gp.disp.y(igp_))
        file.write(fp_disp, ' ')
        file.write(fp_disp, block.gp.disp.z(igp_))
        file.write(fp_disp, 13)
    end_loop
    
    file.close(fp_nodes)
    file.close(fp_disp)
    
    ; 导出单元中心应力
    local fp_stress = file.open('stress.txt', 1, 0)
    
    loop foreach local iz_ block.zone.list
        file.write(fp_stress, block.zone.stress.xx(iz_))
        file.write(fp_stress, ' ')
        file.write(fp_stress, block.zone.stress.yy(iz_))
        file.write(fp_stress, ' ')
        file.write(fp_stress, block.zone.stress.zz(iz_))
        file.write(fp_stress, ' ')
        file.write(fp_stress, block.zone.stress.xy(iz_))
        file.write(fp_stress, ' ')
        file.write(fp_stress, block.zone.stress.xz(iz_))
        file.write(fp_stress, ' ')
        file.write(fp_stress, block.zone.stress.yz(iz_))
        file.write(fp_stress, 13)
    end_loop
    
    file.close(fp_stress)
    
    ; 导出孔边界节点 (半径约为1.0m的节点)
    local fp_hole = file.open('boundary_hole.txt', 1, 0)
    loop foreach local igp_ block.gp.list
        local rad = math.sqrt(block.gp.pos.x(igp_)^2 + block.gp.pos.z(igp_)^2)
        if math.abs(rad - 1.0) < 0.1 then
            file.write(fp_hole, block.gp.pos.x(igp_))
            file.write(fp_hole, ' ')
            file.write(fp_hole, block.gp.pos.y(igp_))
            file.write(fp_hole, ' ')
            file.write(fp_hole, block.gp.pos.z(igp_))
            file.write(fp_hole, 13)
        end_if
    end_loop
    file.close(fp_hole)
    
    ; 导出远场边界节点 (半径约为5.0m的节点)
    local fp_far = file.open('boundary_far.txt', 1, 0)
    loop foreach local igp_ block.gp.list
        local rad = math.sqrt(block.gp.pos.x(igp_)^2 + block.gp.pos.z(igp_)^2)
        if math.abs(rad - 5.0) < 0.1 then
            file.write(fp_far, block.gp.pos.x(igp_))
            file.write(fp_far, ' ')
            file.write(fp_far, block.gp.pos.y(igp_))
            file.write(fp_far, ' ')
            file.write(fp_far, block.gp.pos.z(igp_))
            file.write(fp_far, 13)
        end_if
    end_loop
    file.close(fp_far)
    
    io.out('数据导出完成!')
end

[export_for_pinn]
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fish_script)
    
    print(f"\n已创建3DEC导出脚本: {output_path}")
    print("请在3DEC中运行此脚本以导出数据")


if __name__ == "__main__":
    # 设置路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_path, '../PINNs/main/Data/elastic_hole_3dec.mat')
    
    print("=" * 60)
    print("3DEC数据导出工具 - PINN训练数据准备")
    print("=" * 60)
    
    # 创建FISH导出脚本
    fish_script_path = os.path.join(base_path, 'export_3dec_data.fis')
    create_sample_fish_export_script(fish_script_path)
    
    print("\n" + "=" * 60)
    print("使用步骤:")
    print("=" * 60)
    print("1. 在3DEC中打开模型并运行到收敛")
    print(f"2. 在3DEC中执行: call {fish_script_path}")
    print("3. 确认生成了以下文本文件:")
    print("   - nodes.txt")
    print("   - displacement.txt")
    print("   - stress.txt")
    print("   - boundary_hole.txt")
    print("   - boundary_far.txt")
    print("4. 再次运行此Python脚本完成.mat文件生成")
    print("=" * 60)
    
    # 检查是否存在导出的文本文件
    required_files = ['nodes.txt', 'displacement.txt', 'stress.txt']
    files_exist = all(os.path.exists(os.path.join(base_path, f)) for f in required_files)
    
    if files_exist:
        print("\n检测到3DEC导出文件，开始转换...")
        data = read_3dec_results(base_path)
        export_to_mat(data, output_file)
        print("\n✓ 转换完成！")
    else:
        print("\n⚠ 尚未检测到3DEC导出文件")
        print("   请先在3DEC中运行导出脚本")

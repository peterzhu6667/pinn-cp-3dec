"""
生成圆柱孔弹性力学问题的Kirsch解析解

此脚本基于 ElasticHole.fis 中的解析解公式，生成PINN训练所需的解析解数据。

参考文献:
    - Kirsch, G. (1898). Die Theorie der Elastizität.
    - Jaeger & Cook (1976). Fundamentals of Rock Mechanics.

输出文件: elastic_hole_analytical.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

class KirschSolution:
    """
    Kirsch解析解计算器
    
    根据 ElasticHole.fis 中的公式实现
    """
    
    def __init__(self, p1=30e6, p2=15e6, a=1.0, sm=2.8e9, bm=3.9e9):
        """
        初始化材料参数和边界条件
        
        参数:
            p1: float, 水平远场应力 (Pa) - 对应x方向
            p2: float, 垂直远场应力 (Pa) - 对应y方向  
            a: float, 孔半径 (m)
            sm: float, 剪切模量 G (Pa)
            bm: float, 体积模量 K (Pa)
        """
        self.p1 = p1
        self.p2 = p2
        self.a = a
        self.sm = sm
        self.bm = bm
        
        # 计算泊松比 (来自 ElasticHole.fis)
        self.nu = (3.0 * bm - 2.0 * sm) / (6.0 * bm + 2.0 * sm)
        
        print(f"材料参数:")
        print(f"  剪切模量 G = {sm/1e9:.2f} GPa")
        print(f"  体积模量 K = {bm/1e9:.2f} GPa")
        print(f"  泊松比 ν = {self.nu:.4f}")
        print(f"\n边界条件:")
        print(f"  孔半径 a = {a:.2f} m")
        print(f"  水平应力 σ_x = {p1/1e6:.1f} MPa")
        print(f"  垂直应力 σ_y = {p2/1e6:.1f} MPa")
    
    def stress_radial(self, r, theta):
        """
        径向应力 σ_r (Kirsch解)
        
        来自 ElasticHole.fis 中的 nastr 函数:
        sigre = 0.5*(p1+p2)*(1-o_r2) + 0.5*(p1-p2)*(1-4*o_r2+3*o_r4)*cos(2θ)
        
        参数:
            r: 半径坐标 (m)
            theta: 角度坐标 (弧度)
        
        返回:
            σ_r: 径向应力 (Pa)
        """
        o_r = self.a / r  # a/r
        o_r2 = o_r * o_r  # (a/r)^2
        o_r4 = o_r2 * o_r2  # (a/r)^4
        
        sigr = 0.5 * (self.p1 + self.p2) * (1.0 - o_r2)
        sigr += 0.5 * (self.p1 - self.p2) * (1.0 - 4.0*o_r2 + 3.0*o_r4) * np.cos(2.0 * theta)
        
        return sigr
    
    def stress_tangential(self, r, theta):
        """
        环向应力 σ_θ (Kirsch解)
        
        来自 ElasticHole.fis 中的 nastr 函数:
        sigte = 0.5*(p1+p2)*(1+o_r2) - 0.5*(p1-p2)*(1+3*o_r4)*cos(2θ)
        
        参数:
            r: 半径坐标 (m)
            theta: 角度坐标 (弧度)
        
        返回:
            σ_θ: 环向应力 (Pa)
        """
        o_r = self.a / r
        o_r2 = o_r * o_r
        o_r4 = o_r2 * o_r2
        
        sigt = 0.5 * (self.p1 + self.p2) * (1.0 + o_r2)
        sigt -= 0.5 * (self.p1 - self.p2) * (1.0 + 3.0*o_r4) * np.cos(2.0 * theta)
        
        return sigt
    
    def displacement_radial(self, r, theta):
        """
        径向位移 u_r (Kirsch解)
        
        来自 ElasticHole.fis 中的 nadis 函数:
        dise = 0.25 * (p1+p2 + (p1-p2)*(4*(1-ν) - o_r²)*cos(2θ)) * o_r * a / sm
        
        参数:
            r: 半径坐标 (m)
            theta: 角度坐标 (弧度)
        
        返回:
            u_r: 径向位移 (m)
        """
        o_r = self.a / r
        o_r2 = o_r * o_r
        
        term1 = self.p1 + self.p2
        term2 = (self.p1 - self.p2) * (4.0*(1.0 - self.nu) - o_r2) * np.cos(2.0 * theta)
        
        ur = 0.25 * (term1 + term2) * o_r * self.a / self.sm
        
        return ur
    
    def displacement_tangential(self, r, theta):
        """
        环向位移 u_θ (Kirsch解)
        
        推导自位移场的环向分量
        
        参数:
            r: 半径坐标 (m)
            theta: 角度坐标 (弧度)
        
        返回:
            u_θ: 环向位移 (m)
        """
        o_r = self.a / r
        o_r2 = o_r * o_r
        
        # 环向位移公式
        ut = -0.25 * (self.p1 - self.p2) * (4.0*(1.0 - self.nu) + o_r2) * o_r * self.a / self.sm
        ut *= np.sin(2.0 * theta)
        
        return ut
    
    def cartesian_to_polar(self, x, z):
        """
        笛卡尔坐标转极坐标
        
        参数:
            x, z: 笛卡尔坐标 (m)
        
        返回:
            r, theta: 极坐标 (m, 弧度)
        """
        r = np.sqrt(x**2 + z**2)
        theta = np.arctan2(z, x)
        return r, theta
    
    def polar_to_cartesian_stress(self, sigma_r, sigma_theta, theta):
        """
        极坐标应力转换为笛卡尔坐标应力
        
        参数:
            sigma_r: 径向应力 (Pa)
            sigma_theta: 环向应力 (Pa)
            theta: 角度 (弧度)
        
        返回:
            sigma_xx, sigma_zz, sigma_xz: 笛卡尔坐标应力 (Pa)
        """
        cos2 = np.cos(theta)**2
        sin2 = np.sin(theta)**2
        sin_cos = np.sin(theta) * np.cos(theta)
        
        sigma_xx = sigma_r * cos2 + sigma_theta * sin2
        sigma_zz = sigma_r * sin2 + sigma_theta * cos2
        sigma_xz = (sigma_r - sigma_theta) * sin_cos
        
        return sigma_xx, sigma_zz, sigma_xz
    
    def polar_to_cartesian_displacement(self, u_r, u_theta, theta):
        """
        极坐标位移转换为笛卡尔坐标位移
        
        参数:
            u_r: 径向位移 (m)
            u_theta: 环向位移 (m)
            theta: 角度 (弧度)
        
        返回:
            u_x, u_z: 笛卡尔坐标位移 (m)
        """
        u_x = u_r * np.cos(theta) - u_theta * np.sin(theta)
        u_z = u_r * np.sin(theta) + u_theta * np.cos(theta)
        
        return u_x, u_z


def generate_solution_grid(kirsch, r_min=1.0, r_max=5.0, nr=50, ntheta=100):
    """
    在网格上生成解析解
    
    参数:
        kirsch: KirschSolution实例
        r_min: 最小半径 (m)
        r_max: 最大半径 (m)
        nr: 径向点数
        ntheta: 环向点数
    
    返回:
        dict: 包含所有解析解数据的字典
    """
    # 创建极坐标网格
    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(0, 2*np.pi, ntheta)
    R, Theta = np.meshgrid(r, theta)
    
    # 转换为笛卡尔坐标
    X = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    Y = np.zeros_like(X)  # 2D问题，y=0
    
    # 计算极坐标解
    sigma_r = kirsch.stress_radial(R, Theta)
    sigma_theta = kirsch.stress_tangential(R, Theta)
    u_r = kirsch.displacement_radial(R, Theta)
    u_theta = kirsch.displacement_tangential(R, Theta)
    
    # 转换为笛卡尔坐标
    sigma_xx, sigma_zz, sigma_xz = kirsch.polar_to_cartesian_stress(sigma_r, sigma_theta, Theta)
    u_x, u_z = kirsch.polar_to_cartesian_displacement(u_r, u_theta, Theta)
    
    # 准备输出数据
    data = {
        # 坐标 (展平为1D数组)
        'x': X.flatten(),
        'y': Y.flatten(),
        'z': Z.flatten(),
        
        # 极坐标
        'r': R.flatten(),
        'theta': Theta.flatten(),
        
        # 极坐标应力
        'sigma_r': sigma_r.flatten(),
        'sigma_theta': sigma_theta.flatten(),
        
        # 笛卡尔坐标应力
        'sigma_xx': sigma_xx.flatten(),
        'sigma_yy': sigma_xx.flatten(),  # 平面应变，σ_yy = σ_xx
        'sigma_zz': sigma_zz.flatten(),
        'sigma_xy': np.zeros(X.size),
        'sigma_xz': sigma_xz.flatten(),
        'sigma_yz': np.zeros(X.size),
        
        # 极坐标位移
        'u_r': u_r.flatten(),
        'u_theta': u_theta.flatten(),
        
        # 笛卡尔坐标位移
        'u_x': u_x.flatten(),
        'u_y': np.zeros(X.size),  # 2D问题
        'u_z': u_z.flatten(),
        
        # 网格参数
        'nr': np.array([nr]),
        'ntheta': np.array([ntheta]),
        'r_min': np.array([r_min]),
        'r_max': np.array([r_max]),
    }
    
    print(f"\n生成解析解网格:")
    print(f"  径向: {r_min:.2f} - {r_max:.2f} m, {nr} 点")
    print(f"  环向: 0 - 2π, {ntheta} 点")
    print(f"  总点数: {X.size}")
    
    return data


def plot_analytical_solution(data, kirsch):
    """
    可视化解析解
    """
    nr = data['nr'][0]
    ntheta = data['ntheta'][0]
    
    X = data['x'].reshape(ntheta, nr)
    Z = data['z'].reshape(ntheta, nr)
    sigma_r = data['sigma_r'].reshape(ntheta, nr)
    sigma_theta = data['sigma_theta'].reshape(ntheta, nr)
    u_r = data['u_r'].reshape(ntheta, nr)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 径向应力
    c1 = axes[0].contourf(X, Z, sigma_r/1e6, levels=20, cmap='RdBu_r')
    axes[0].set_title('Radial Stress $\\sigma_r$ (MPa)')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('z (m)')
    axes[0].axis('equal')
    plt.colorbar(c1, ax=axes[0])
    
    # 环向应力
    c2 = axes[1].contourf(X, Z, sigma_theta/1e6, levels=20, cmap='RdBu_r')
    axes[1].set_title('Tangential Stress $\\sigma_\\theta$ (MPa)')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')
    axes[1].axis('equal')
    plt.colorbar(c2, ax=axes[1])
    
    # 径向位移
    c3 = axes[2].contourf(X, Z, u_r*1000, levels=20, cmap='viridis')
    axes[2].set_title('Radial Displacement $u_r$ (mm)')
    axes[2].set_xlabel('x (m)')
    axes[2].set_ylabel('z (m)')
    axes[2].axis('equal')
    plt.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(__file__), 'analytical_solution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图片已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Kirsch解析解生成器 - 圆柱孔弹性力学问题")
    print("=" * 70)
    
    # 创建Kirsch解算器 (参数来自 ElasticHole.fis)
    kirsch = KirschSolution(
        p1=30e6,   # 水平远场应力 (Pa)
        p2=15e6,   # 垂直远场应力 (Pa)
        a=1.0,     # 孔半径 (m)
        sm=2.8e9,  # 剪切模量 (Pa)
        bm=3.9e9   # 体积模量 (Pa)
    )
    
    # 生成解析解数据
    data = generate_solution_grid(kirsch, r_min=1.0, r_max=5.0, nr=50, ntheta=100)
    
    # 添加材料参数
    data['shear_modulus'] = np.array([kirsch.sm])
    data['bulk_modulus'] = np.array([kirsch.bm])
    data['poissons_ratio'] = np.array([kirsch.nu])
    data['hole_radius'] = np.array([kirsch.a])
    data['far_field_stress_x'] = np.array([kirsch.p1])
    data['far_field_stress_y'] = np.array([kirsch.p2])
    
    # 保存为.mat文件
    output_file = os.path.join(
        os.path.dirname(__file__), 
        '../PINNs/main/Data/elastic_hole_analytical.mat'
    )
    sio.savemat(output_file, data)
    
    print("\n" + "=" * 70)
    print(f"✓ 解析解数据已导出到: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
    print("=" * 70)
    
    # 可视化
    plot_analytical_solution(data, kirsch)
    
    print("\n数据字典包含的键:")
    for key in sorted(data.keys()):
        if isinstance(data[key], np.ndarray):
            print(f"  {key:25s}: shape {data[key].shape}")

"""本项目分析图统一使用简体中文，并检查字体字形。"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager,ft2font

def configure(text='训练验证策略价值专家候选监督学习曲线旧能力保留误差更新次数'):
    for name in ('Hiragino Sans GB','Noto Sans CJK SC','Songti SC'):
        try:path=font_manager.findfont(font_manager.FontProperties(family=name),fallback_to_default=False)
        except ValueError:continue
        chars=ft2font.FT2Font(path).get_charmap()
        if all(ord(c) in chars for c in text if not c.isspace()):
            matplotlib.rcParams.update({'font.family':name,'axes.unicode_minus':False,'svg.fonttype':'path'})
            return path
    raise RuntimeError('没有包含所需简体中文字形的字体，拒绝生成乱码图表')

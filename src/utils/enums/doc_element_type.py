import enum

class SpanType(enum.Enum):
    # 文档结构元素
    Title = 'title'
    Header = 'header'
    Section = 'section'
    SubSection = 'sub_section'
    Paragraph = 'paragraph'
    Text = 'text'
    List = 'list'
    
    # 表格相关元素
    Table = 'table'
    TableCaption = 'table_caption'
    TableFootnote = 'table_footnote'
    
    # 图像相关元素
    Image = 'image'
    ImageCaption = 'image_caption'
    ImageFootnote = 'image_footnote'
    
    # 公式相关元素
    Formula = 'formula'
    InlineEquation = 'inline_equation'
    
    # 布局辅助元素
    Separator = 'separator'
    Footer = 'footer'
    
    # 特殊元素
    Watermark = 'watermark'
    Seal = 'seal'
    Algorithm = 'Algorithm'
    
    # 系统元素
    Discarded = 'discarded'
    Unknown = 'unknown'

class ContentType(enum.Enum):
    Table = 'table'
    Text = 'text'
    Figure = 'figure'
    Unknown = 'unknown'

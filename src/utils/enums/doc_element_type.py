import enum

class BlockType(enum.Enum):
    Image = 'image'
    Table = 'table'
    Text = 'text'
    Title = 'title'
    Paragraph = 'paragraph'
    Footer = 'footer'
    Discarded = 'discarded'
    List = 'list'
    Header = 'header'
    Formula = "formula"
    Separator = 'separator'
    Seal = 'seal'
    Watermark = 'watermark'
    Section = 'Section'
    SubSection = 'SubSection'
    Algorithm = 'Algorithm'
    Unknown = 'unknown'

class ContentType(enum.Enum):
    Table = 'table'
    Text = 'text'
    Figure = 'figure'
    Unknown = 'unknown'

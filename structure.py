# -*- coding: utf-8 -*-
# +
from pathlib import Path


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(
            path for path in root.iterdir() if criteria(path)),
                          key=lambda s: str(s).lower())
        if len(children) > 0:
            for ext in ['.jpg', '.JPG', '.JPEG', '.jpeg', '.png', '.dcm']:
                if str(children[0]).endswith(ext):
                    children = [Path('*' + ext)]
            if len(children) > 100 and children[0].is_dir():
                children = [Path(f'[{len(children)} folders]')]
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle if parent.
                         is_last else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


# -

paths = DisplayablePath.make_tree(Path('dataset'))

for path in paths:
    out = path.displayable()
    if any([x in out for x in ['ipynb_checkpoints', 'gitignore']]):
        continue
    print(path.displayable())

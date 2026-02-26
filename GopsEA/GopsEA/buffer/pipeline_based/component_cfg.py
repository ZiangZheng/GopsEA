from GopsEA import configclass
from typing import List
from dataclasses import MISSING


@configclass
class ComponentCfg:
    comp_names: List[str] = MISSING
    comp_dtype: List[str] = MISSING
    comp_shape: List[int] = MISSING
    
    def get_dtype(self, name):
        idx = self.comp_names.index(name)
        return self.comp_dtype[idx]
    
    def get_shape(self, **mapping):
        shape = [mapping.get(name, self.comp_shape[idx]) for idx, name in enumerate(self.comp_names)]
        assert min(shape) > 0, f"Invalid shape with {shape}"
        return shape
    
    def __str__(self) -> str:
        """
        Human-readable tabular summary of the Component configuration.
        Displays component name, dtype, and shape in aligned columns.
        """
        header = ["Name", "DType", "Shape"]
        rows = list(zip(self.comp_names, self.comp_dtype, self.comp_shape))

        # compute alignment width
        col_widths = [
            max(len(str(x)) for x in col)
            for col in zip(*([header] + rows))
        ]

        def fmt_row(row):
            return "  ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths))

        lines = []
        lines.append("ComponentCfg:")
        lines.append("  " + fmt_row(header))
        lines.append("  " + "  ".join("-" * w for w in col_widths))

        for r in rows:
            lines.append("  " + fmt_row(r))

        return "\n".join(lines)
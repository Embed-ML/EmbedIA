# embedia/tools/extract_functions.py

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    name: str
    full_text: str
    start_line: int
    end_line: int


class SimpleCFunctionExtractor:

    C_KEYWORDS = {
        'if', 'else', 'for', 'while', 'do', 'switch', 'case',
        'return', 'break', 'continue', 'goto', 'sizeof', 'typedef',
        'struct', 'union', 'enum', 'static', 'const', 'volatile',
        'extern', 'auto', 'register', 'inline', 'void'
    }

    CONTROL_KEYWORDS = ('return', 'if', 'for', 'while', 'switch', 'case')

    def __init__(self, min_lines: int = 3):
        self.min_lines = min_lines
        self.functions = []

    def extract_from_file(self, filepath: Path) -> List[FunctionInfo]:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.functions = []
        i = 0

        while i < len(lines):
            line = lines[i].rstrip('\n')
            stripped = line.strip()

            # Ignorar líneas vacías y triviales
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                i += 1
                continue

            # Ignorar comentarios multilinea
            if stripped.startswith('/*'):
                while i < len(lines):
                    if '*/' in lines[i]:
                        i += 1
                        break
                    i += 1
                continue

            # Ignorar estructuras de control
            if any(stripped.startswith(k) for k in self.CONTROL_KEYWORDS):
                i += 1
                continue

            # Debe tener '(' pero no ser prototipo
            if '(' in line and not stripped.endswith(';'):

                # Lookahead estructural
                lookahead_lines = lines[i:i+6]
                found_brace = False

                for la in lookahead_lines:
                    la_strip = la.strip()

                    if not la_strip:
                        continue

                    if ';' in la_strip:
                        break

                    if '{' in la_strip:
                        found_brace = True
                        break

                if not found_brace:
                    i += 1
                    continue

                func = self._try_extract_function(lines, i)

                if func:
                    self.functions.append(func)
                    i = func.end_line + 1
                    continue

            i += 1

        return self._deduplicate(self.functions)

    def _try_extract_function(self, lines: List[str], start_idx: int) -> Optional[FunctionInfo]:

        signature_lines = []
        open_brace_line = None

        # Construir firma multilinea
        for offset in range(10):
            if start_idx + offset >= len(lines):
                return None

            line = lines[start_idx + offset].strip()

            if not line or line.startswith('//') or line.startswith('#'):
                continue

            signature_lines.append(line)

            if '{' in line:
                open_brace_line = start_idx + offset
                break

        if open_brace_line is None:
            return None

        signature = ' '.join(signature_lines)

        if '(' not in signature or ')' not in signature:
            return None

        before_paren = signature[:signature.index('(')].strip()

        tokens = before_paren.split()
        if len(tokens) < 2:
            return None

        # Extraer nombre con regex
        match = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*$', before_paren)
        if not match:
            return None

        func_name = match.group(1)

        if func_name in self.C_KEYWORDS:
            return None

        # Buscar cierre
        end_line = self._find_closing_brace(lines, open_brace_line)
        if end_line is None:
            return None

        if end_line - start_idx + 1 < self.min_lines:
            return None

        full_text = ''.join(lines[start_idx:end_line + 1])

        return FunctionInfo(
            name=func_name,
            full_text=full_text,
            start_line=start_idx + 1,
            end_line=end_line + 1
        )

    def _find_closing_brace(self, lines: List[str], start_line: int) -> Optional[int]:
        brace_count = 0
        in_string = False
        in_char = False
        in_comment = False

        for i in range(start_line, len(lines)):
            line = lines[i]
            j = 0

            while j < len(line):
                char = line[j]

                if char == '"' and not in_char and not in_comment:
                    if j == 0 or line[j - 1] != '\\':
                        in_string = not in_string

                elif char == "'" and not in_string and not in_comment:
                    if j == 0 or line[j - 1] != '\\':
                        in_char = not in_char

                elif char == '/' and j + 1 < len(line) and not in_string and not in_char:
                    if line[j + 1] == '*':
                        in_comment = True
                        j += 1
                    elif line[j + 1] == '/':
                        break

                elif char == '*' and j + 1 < len(line) and in_comment:
                    if line[j + 1] == '/':
                        in_comment = False
                        j += 1

                elif not in_string and not in_char and not in_comment:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return i

                j += 1

        return None

    def _deduplicate(self, functions: List[FunctionInfo]) -> List[FunctionInfo]:
        seen = set()
        result = []

        for f in functions:
            key = (f.name, f.start_line)
            if key not in seen:
                seen.add(key)
                result.append(f)

        return result


def write_function_files(functions: List[FunctionInfo], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for func in functions:
        filename = f"{func.name}.c"
        filepath = output_dir / filename

        header = f"""/**
 * Function: {func.name}
 * Lines: {func.start_line}-{func.end_line}
 */

"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + func.full_text)

        print(f"✓ Created: {filename}")


def rewrite_with_includes(
    input_file: Path,
    functions: List[FunctionInfo],
    output_dir: Path,
):
    """
    Crea <input_file>.bak con el contenido original y reescribe
    <input_file> reemplazando cada función detectada por un comentario
    // @embedia-include <output_dir_name>/<func_name>.c
    """
    # 1. Backup
    bak_path = input_file.with_suffix(input_file.suffix + '.bak')
    shutil.copy2(input_file, bak_path)
    print(f"✓ Backup created: {bak_path.name}")

    # 2. Leer líneas originales (1-indexed internamente)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Construir set de rangos a suprimir: {line_index_0based}
    # Cada función ocupa [start_line-1 .. end_line-1] (inclusive, 1-based stored)
    suppressed: dict[int, Optional[str]] = {}  # line_0based -> include str or None

    # Path del include relativo al output_dir
    # Ej: output_dir = /project/neural_net  → include_prefix = "neural_net"
    # Ej: output_dir = /project/src/impl    → include_prefix = "src/impl"
    try:
        include_prefix = output_dir.resolve().relative_to(output_dir.resolve().parent)
    except ValueError:
        include_prefix = Path(output_dir.name)
    include_prefix = str(include_prefix).replace("\\", "/")

    for func in functions:
        start_0 = func.start_line - 1
        end_0 = func.end_line - 1
        include_comment = f"// @embedia-include {include_prefix}/{func.name}.c\n"
        for idx in range(start_0, end_0 + 1):
            # Primera línea del rango → include; resto → suprimir
            suppressed[idx] = include_comment if idx == start_0 else None

    # 3. Reconstruir archivo
    new_lines = []
    for idx, line in enumerate(lines):
        if idx in suppressed:
            replacement = suppressed[idx]
            if replacement is not None:
                new_lines.append(replacement)
            # None → línea suprimida (resto del cuerpo de la función)
        else:
            new_lines.append(line)

    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✓ Rewritten: {input_file.name}")


# ================= CLI =================

class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Extrae funciones de un archivo C.',
        epilog=(
            "Ejemplos:\n"
            "  python extract_functions.py main.c\n"
            "  python extract_functions.py main.c -o impl/\n"
            "  python extract_functions.py main.c --dry-run\n"
        ),
        formatter_class=CustomFormatter
    )

    parser.add_argument('input_file', type=Path, help='Archivo C de entrada')
    parser.add_argument('--output-dir', '-o', type=Path, help='Directorio de salida')
    parser.add_argument('--min-lines', type=int, default=3, help='Líneas mínimas')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Simulación')

    args = parser.parse_args()

    extractor = SimpleCFunctionExtractor(args.min_lines)
    funcs = extractor.extract_from_file(args.input_file)

    print(f"\nFound {len(funcs)} functions:\n")
    for f in funcs:
        print(f" - {f.name} ({f.start_line}-{f.end_line})")

    if not funcs:
        print("\nNo functions found")
        return 0

    # Definir output dir si no viene
    output_dir = args.output_dir
    if not output_dir:
        output_dir = args.input_file.parent / (args.input_file.stem + '_impl')

    print(f"\nWriting files to: {output_dir}\n")

    if args.dry_run:
        print("(dry-run: no files written)")
        return 0

    write_function_files(funcs, output_dir)
    rewrite_with_includes(args.input_file, funcs, output_dir)


if __name__ == '__main__':
    import sys
    sys.exit(main())
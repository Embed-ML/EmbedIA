from embedia.utils.binary_helper import BinaryGlobalMask
from pathlib import Path
import re

def declare_array(dt_type, var_name, dt_conv, data_array, limit=80):

    if dt_conv is None:
        dt_conv = lambda x:x
    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            line += f'''{dt_conv(v)}, '''
            if len(line) >= limit:
                val += line + '\n    '
                line = ''
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code


def declare_array2(toti,xBits,lista_contadores,dt_type, var_name, dt_conv, data_array, limit=80):

    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            
            lista_contadores[2] = lista_contadores[2] + 1
            
            if xBits==16:
                if v == 1.0:  
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_16())[lista_contadores[1]]
            elif xBits==32:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_32())[lista_contadores[1]]
            elif xBits==64:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_64())[lista_contadores[1]]
            else:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_8())[lista_contadores[1]]
            
            if lista_contadores[1] == xBits-1 or (lista_contadores[2] == toti):
                
                line += f'''{dt_conv(lista_contadores[0])}, '''
                if len(line) >= limit:
                    val += line + '\n    '
                    line = ''
                lista_contadores[1] = 0
                lista_contadores[0] = 0
                
            else:
                lista_contadores[1] = lista_contadores[1] +1
            
            
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code



def replace_c_define(content, values):
    """
    Reemplaza valores de constantes definidas con #define en código C/C++.

    Args:
        content (str): El contenido del código fuente donde buscar los #define
        values: Puede ser:
            - Una tupla (nombre_constante, valor_constante)
            - Una lista de tuplas [(nombre1, valor1), (nombre2, valor2), ...]
            Los valores pueden ser: int, float, str, bool

    Returns:
        str: El contenido con los valores de #define reemplazados

    Examples:
        >>> code = "#define PI 3.14\\n#define MAX_SIZE 100"
        >>> replace_c_define(code, ("PI", 3.14159))  # float
        '#define PI 3.14159\\n#define MAX_SIZE 100'

        >>> replace_c_define(code, [("PI", 3.14159), ("MAX_SIZE", 200)])  # mixed types
        '#define PI 3.14159\\n#define MAX_SIZE 200'

        >>> replace_c_define(code, ("DEBUG", True))  # bool -> "True"
        >>> replace_c_define(code, ("FLAG", False))  # bool -> "False"
    """

    # Normalizar la entrada: convertir tupla individual a lista
    if isinstance(values, tuple):
        values = [values]

    # Crear una copia del contenido para modificar
    result = content

    # Procesar cada constante
    for name, value in values:
        # Patrón regex para encontrar #define NOMBRE valor
        # Captura espacios/tabs después de #define y después del nombre
        pattern = r'(#define\s+' + re.escape(name) + r'\s+)([^\s\n]*)'

        # Reemplazar manteniendo la estructura original
        replacement = r'\g<1>' + str(value)
        result = re.sub(pattern, replacement, result)

    return result


class BlockContext:
    """Context manager for code blocks with automatic indentation."""

    def __init__(self, builder, header, footer="}"):
        self.builder = builder
        self.header = header
        self.footer = footer

    def __enter__(self):
        self.builder.add(self.header).inc()
        return self.builder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.builder.dec().add(self.footer)
        return False


class CBuilder:
    """
    Text builder with indentation support for C code generation.

    Design principles:
    - All qualifiers (static, const, EMBEDIA_MODEL_STORAGE, etc.) are the caller's
      responsibility — methods never assume or inject qualifiers.
    - Methods return self for chaining: cb.add('x').inc()
    - bgn() handles braces and indentation automatically.
    - add_array() and add_struct() are semantic helpers for the most common
      C patterns in EmbedIA layer initialization.
    """

    def __init__(self, indent_size=4):
        self.indent_size = indent_size
        self._current_indent = 0
        self._lines = []

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def add(self, text=""):
        """Add a line with current indentation. Empty text adds a blank line."""
        indent = ' ' * self._current_indent
        if text:
            self._lines.append(indent + text.replace('\n', '\n' + indent))
        else:
            self._lines.append('')
        return self

    def append(self, text=""):
        """Alias for add()."""
        return self.add(text)

    def inc(self):
        """Increase indentation by one level."""
        self._current_indent += self.indent_size
        return self

    def dec(self):
        """Decrease indentation by one level (floor at 0)."""
        self._current_indent = max(0, self._current_indent - self.indent_size)
        return self

    def bgn(self, header, footer="}"):
        """
        Context manager for a C block.

        Appends ' {' to the header automatically, increases indentation
        inside the block, and closes with footer (default '}') on exit.

        Usage:
            with cb.bgn('void foo(void)'):
                cb.add('return;')
            # generates:
            # void foo(void) {
            #     return;
            # }

        If header already ends with '{', no extra '{' is added — this
        allows callers to pass custom openers like 'do {' or 'struct {'.
        """
        opener = header if header.rstrip().endswith('{') else header + ' {'
        return BlockContext(self, opener, footer)

    def end(self, footer=""):
        """Manually close a block (dec + optional footer line)."""
        self.dec()
        if footer:
            self.add(footer)
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_code(self):
        """Return generated code as a single string."""
        return '\n'.join(self._lines)

    def __str__(self):
        return self.get_code()

    def clear(self):
        """Reset builder to empty state."""
        self._lines = []
        self._current_indent = 0
        return self

    def load(self, filename):
        """Replace current content with the contents of a file."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File '{filename}' not found")
        self.clear()
        for line in path.read_text(encoding='utf-8').splitlines():
            self._lines.append(line)
        return self

    def save(self, filename):
        """Write current content to a file (creates parent dirs if needed)."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.get_code(), encoding='utf-8')
        return self

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def to_array(self, values, sep=', ', fmt=''):
        """Format an iterable as a comma-separated string."""
        result = []
        for x in values:
            try:
                result.append(f"{x: {fmt}}")
            except (ValueError, TypeError):
                result.append(str(x))
        return sep.join(result)

    def indent_text(self, text, times=1, char=' '):
        """Return text with extra indentation (does not add to builder)."""
        pad = char * (self.indent_size * times)
        return pad + text.replace('\n', '\n' + pad)

    def printf(self, format_string, *args):
        """Add a printf() call."""
        if args:
            self.add(f'printf("{format_string}", {", ".join(str(a) for a in args)});')
        else:
            self.add(f'printf("{format_string}");')
        return self

    # ------------------------------------------------------------------
    # Semantic C helpers
    # ------------------------------------------------------------------

    def add_array(self, dtype, name, values, cols=0, comments=None, line_limit=80, header_comment=''):
        """
        Generate a C array declaration.

        The caller supplies the full type string including all qualifiers:

            cb.add_array('static EMBEDIA_MODEL_STORAGE fixed', 'weights0',
                         values, cols=4, comments=original_rows)

            cb.add_array('static filter_t', 'filters', filter_inits)

        Parameters
        ----------
        dtype      : full type string (qualifiers + base type)
        name       : variable name (None / '' → anonymous initializer only)
        values     : iterable — any iterable is accepted: list, numpy array, generator.
                     Callers do not need to call .tolist() before passing numpy arrays.
        cols       : elements per row.
                     > 0 → fixed column layout, one row per cols elements.
                     0   → free layout, wrap at line_limit characters.
        comments   : list of per-row comment strings (optional).
                     When cols > 0: one comment per row of cols elements.
                     When cols = 0: one comment per wrapped line.
        line_limit : max line length for free layout (cols=0). Default 80.
        """
        values = list(values)

        comment = f'  // {header_comment}' if header_comment else ''
        opener = f'{dtype} {name}[] = {{{comment}' if name else f'{{{comment}'
        self.add(opener).inc()

        if cols > 0:
            # fixed column layout — one row per cols elements
            rows = [values[i:i + cols] for i in range(0, len(values), cols)]
            for row_idx, row in enumerate(rows):
                is_last = (row_idx == len(rows) - 1)
                row_str = ', '.join(str(v) for v in row)
                trailing = '' if is_last else ','
                comment = f'  /* {comments[row_idx]} */' \
                    if comments and row_idx < len(comments) else ''
                self.add(f'{row_str}{trailing}{comment}')

        else:
            # free layout — wrap lines at line_limit characters
            current_line = ''
            wrapped_lines = []
            for idx, v in enumerate(values):
                is_last = (idx == len(values) - 1)
                token = str(v) + ('' if is_last else ', ')
                if current_line and len(current_line) + len(token) > line_limit:
                    wrapped_lines.append(current_line.rstrip(', '))
                    current_line = token
                else:
                    current_line += token
            if current_line:
                wrapped_lines.append(current_line.rstrip(', '))

            for line_idx, line in enumerate(wrapped_lines):
                is_last = (line_idx == len(wrapped_lines) - 1)
                trailing = '' if is_last else ','
                comment = f'  /* {comments[line_idx]} */' \
                    if comments and line_idx < len(comments) else ''
                self.add(f'{line}{trailing}{comment}')

        self.dec().add('};')
        return self

    def add_struct(self, dtype, name, fields, comments=None):
        """
        Generate a C struct initialization.

        Each element of fields becomes one indented line inside the braces,
        so the caller controls grouping — pass multi-value strings to put
        related fields on the same line:

            # one field per line
            cb.add_struct('static conv2d_layer_t', 'layer',
                          ['8', 'filters', '1', '{2,2}', '0', '{1,1}'])

            # grouped by meaning — compact and readable
            cb.add_struct('static EMBEDIA_MODEL_STORAGE conv2d_layer_t', 'layer', [
                f'{n_filters}, filters, {n_channels}',  # dimensions
                f'{kernel_size}, {padding}, {strides}'  # geometry
            ])

        Parameters
        ----------
        dtype    : full type string including qualifiers
        name     : variable name
        fields   : list of strings — each becomes one indented line
        comments : optional per-field comment strings
        """
        self.add(f'{dtype} {name} = {{').inc()

        for idx, field in enumerate(fields):
            is_last = (idx == len(fields) - 1)
            trailing = '' if is_last else ','
            comment = f'  /* {comments[idx]} */' \
                if comments and idx < len(comments) else ''

            if field+comment != '':
                self.add(f'{field}{trailing}{comment}')

        self.dec().add('};')
        return self


class ArduinoBuilder(CBuilder):
    """Arduino-specific code builder with Serial.print printf implementation."""

    def printf(self, format_string, *args):
        """Add printf-like statements using Serial.print for Arduino."""
        # Improved pattern to capture:
        # % - (optional) flags (-, +, space, 0)
        # (optional) width
        # (optional) .precision
        # conversion type
        pattern = r'%([-+ 0]*)(\d*)(?:\.(\d+))?([dfsuoxX])'

        matches = list(re.finditer(pattern, format_string))
        if len(matches) != len(args):
            raise ValueError(f"Number of format specifiers ({len(matches)}) doesn't match arguments ({len(args)})")

        last_end = 0

        for i, match in enumerate(matches):
            # Add text before the specifier
            if match.start() > last_end:
                text = format_string[last_end:match.start()]
                self.append(f'Serial.print("{text}");')

            # Process the specifier
            flags = match.group(1)
            width = match.group(2)
            precision = match.group(3)
            type_spec = match.group(4)
            arg = args[i]

            # Debug print
            #print(f"Processing: {match.group(0)}")
            #print(f"Flags: '{flags}', Width: '{width}', Precision: '{precision}', Type: '{type_spec}'")

            if type_spec == 's':
                # String handling (width for strings would need manual padding in Arduino)
                self.append(f'Serial.print("{arg}");')
            elif type_spec == 'f':
                # Float handling
                decimals = precision if precision else '2'
                self.append(f'Serial.print({arg}, {decimals});')
            elif type_spec in ('d', 'i', 'u', 'o', 'x', 'X'):
                # Integer types (width and zero-padding would need manual handling)
                self.append(f'Serial.print({arg});')
            else:
                self.append(f'Serial.print({arg});')

            last_end = match.end()

        # Add remaining text
        if last_end < len(format_string):
            text = format_string[last_end:]
            self.append(f'Serial.print("{text}");')






import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

def save_to_file(filename, content):
    file = open(filename, 'w', encoding='utf-8')
    file.write(content)
    file.close()
    
def read_from_file(filename):
    file = open(filename, 'r', encoding='utf-8')
    # content = file.readlines()
    content = file.read()
    file.close()  
    return content


def copy(src_file, dst_file):
    shutil.copy(src_file, dst_file)


def resolve_link_file(file_path, base_folder=None, max_depth=5):
    """
    Resuelve archivos .lnk recursivamente.
    
    Si el archivo existe, lo retorna directamente.
    Si no existe, busca archivo.lnk que contiene la ruta al archivo real.
    Soporta resolución recursiva de múltiples niveles de links.
    
    Args:
        file_path: Ruta del archivo a resolver
        base_folder: Carpeta base para resolver rutas relativas (opcional)
        max_depth: Profundidad máxima de resolución de links (default: 5)
    
    Returns:
        str: Ruta del archivo real resuelto
    
    Raises:
        FileNotFoundError: Si el archivo no existe y no hay .lnk
        ValueError: Si se excede la profundidad máxima de links
    """
    depth = 0
    current_path = file_path
    
    while depth < max_depth:
        # Si el archivo existe, retornarlo
        if os.path.exists(current_path):
            return current_path
        
        # Buscar archivo.lnk
        link_file = current_path + '.lnk'
        if os.path.exists(link_file):
            # Leer la ruta del archivo real
            target = read_from_file(link_file).strip()
            
            # Eliminar comentarios (líneas que empiezan con #)
            if '#' in target:
                target = target.split('#')[0].strip()
            
            # Resolver ruta relativa o absoluta
            if os.path.isabs(target):
                # Ruta absoluta
                current_path = target
            elif base_folder:
                # Ruta relativa desde base_folder
                current_path = os.path.normpath(os.path.join(base_folder, target))
            else:
                # Ruta relativa desde el directorio del link_file
                link_dir = os.path.dirname(os.path.abspath(link_file))
                current_path = os.path.normpath(os.path.join(link_dir, target))
            
            depth += 1
        else:
            # No existe ni el archivo ni el .lnk
            break
    
    # Validar resultado
    if depth >= max_depth:
        raise ValueError(
            f'Link resolution exceeded max depth ({max_depth}) for: {file_path}'
        )
    
    if not os.path.exists(current_path):
        raise FileNotFoundError(
            f'File not found: {file_path} '
            f'(resolved to: {current_path}, but it does not exist)'
        )
    
    return current_path


def copy_with_link(src_file, dst_file, src_folder=None, max_depth=5):
    """
    Copia un archivo, con soporte para archivos .lnk como redirección.
    
    Si el archivo no existe, busca archivo.lnk que contiene la ruta al archivo real.
    Soporta resolución recursiva de múltiples niveles de links.
    
    Args:
        src_file: Ruta del archivo fuente a copiar
        dst_file: Ruta del archivo destino
        src_folder: Carpeta base para resolver rutas relativas (opcional)
        max_depth: Profundidad máxima de resolución de links (default: 5)
    
    Raises:
        FileNotFoundError: Si el archivo no existe y no hay .lnk válido
        ValueError: Si se excede la profundidad máxima de links
    """
    # Resolver el archivo real (puede ser a través de links)
    actual_src = resolve_link_file(src_file, src_folder, max_depth)
    
    # Copiar el archivo real
    shutil.copy(actual_src, dst_file)


# embedia/model_generator/directive_processor.py

from pathlib import Path
from typing import Set, Optional

from pathlib import Path
from typing import Set, Optional
import os


class DirectiveProcessor:
    """
    Procesa directivas @embedia-include en archivos C.

    Compatible con FileProcessor - puede trabajar con contenido en memoria.

    Sintaxis:
        // @embedia-include-if LayerName path/to/file.c
        // @embedia-include path/to/file.c
    """

    INCLUDE_IF = "// @embedia-include-if "
    INCLUDE_ALWAYS = "// @embedia-include "
    NOTES = "@embedia-note"

    def __init__(self, model, options, search_paths=None):
        """
        Args:
            model: Modelo embedia
            options: Opciones del proyecto
            search_paths: Lista de carpetas donde buscar archivos incluidos
                         Ejemplo: [src_folder, tmpl_folder]
        """
        self.model = model
        self.options = options
        self.search_paths = search_paths or []
        self._available_layers = None
        self._processed_files_cache = {}

    @property
    def available_layers(self) -> Set[str]:
        """Detecta capas disponibles en el modelo (lazy load)"""
        if self._available_layers is None:
            self._available_layers = self._detect_layers()
        return self._available_layers

    def _detect_layers(self) -> Set[str]:
        """Detecta qué tipos de capas usa el modelo"""
        from embedia.core.dummy_layer import DummyLayer
        from embedia.core.unimplemented_layer import UnimplementedLayer

        layers = set()
        for layer in self.model.embedia_layers:
            if isinstance(layer, (DummyLayer, UnimplementedLayer)):
                continue
            layers.add(layer.__class__.__name__)
        return layers

    # ===== MÉTODOS PRINCIPALES (compatibles con FileProcessor) =====

    def process_content(self, content: str, base_file_path: Path) -> str:
        """
        Procesa contenido con directivas ya cargado en memoria.

        Args:
            content: Contenido del archivo (string)
            base_file_path: Path del archivo base para rutas relativas

        Returns:
            Contenido procesado con archivos incluidos
        """
        # Primero eliminar bloques de comentarios con @embedia-note
        content = self._remove_notes_blocks(content)
        
        output_lines = []

        # Procesar línea por línea
        for line in content.splitlines(keepends=True):
            processed_line = self._process_line(line, base_file_path)
            output_lines.append(processed_line)

        return ''.join(output_lines)

    def process_file(self, src_file: Path) -> str:
        """
        Procesa un archivo con directivas (mantenido para compatibilidad).

        Args:
            src_file: Archivo fuente con directivas

        Returns:
            Contenido procesado
        """
        # Usar cache para evitar reprocesar el mismo archivo
        cache_key = str(src_file.resolve())
        if cache_key in self._processed_files_cache:
            return self._processed_files_cache[cache_key]

        with open(src_file, 'r', encoding='utf-8') as f:
            content = f.read()

        processed_content = self.process_content(content, src_file)

        # Guardar en cache
        self._processed_files_cache[cache_key] = processed_content

        return processed_content

    # ===== MÉTODOS INTERNOS DE PROCESAMIENTO =====

    def _process_line(self, line: str, base_file: Path) -> str:
        """Procesa una línea del archivo"""
        stripped = line.strip()

        # Eliminar comentarios de línea con @embedia-note
        if self.NOTES in line and ('//' in line):
            return ''

        # Verificar si es directiva include condicional
        if stripped.startswith(self.INCLUDE_IF):
            return self._process_include_if(line, base_file)

        # Verificar si es directiva include siempre
        elif stripped.startswith(self.INCLUDE_ALWAYS):
            return self._process_include_always(line, base_file)

        # Línea normal, retornar sin cambios
        return line

    def _process_include_if(self, line: str, base_file: Path) -> str:
        """
        Procesa: // @embedia-include-if LayerName path/to/file.c
        """
        # Extraer partes: LayerName y path
        content = line.strip()[len(self.INCLUDE_IF):].strip()
        parts = content.split(maxsplit=1)

        if len(parts) != 2:
            return f'// ERROR: Invalid directive: {line}'

        layer_name, file_path = parts

        # Verificar si la capa está en el modelo
        if layer_name in self.available_layers:
            return self._include_file(file_path, base_file)
        else:
            return f'// Skipped: {file_path} (layer {layer_name} not used)\n'

    def _process_include_always(self, line: str, base_file: Path) -> str:
        """
        Procesa: // @embedia-include path/to/file.c
        """
        # Extraer path
        file_path = line.strip()[len(self.INCLUDE_ALWAYS):].strip()

        if not file_path:
            return f'// ERROR: Invalid directive: {line}'

        return self._include_file(file_path, base_file)

    def _include_file(self, file_path: str, base_file: Path) -> str:
        """Lee e incluye el contenido de un archivo"""

        resolved_target = None

        # 1. Buscar en search_paths (búsqueda simple)
        if self.search_paths:
            for search_path in self.search_paths:
                candidate = Path(search_path) / file_path
                resolved = self._resolve_file_path(candidate)
                if resolved:
                    resolved_target = resolved
                    break

        # 2. Si no, buscar relativo al archivo base
        if not resolved_target:
            candidate = base_file.parent / file_path
            resolved_target = self._resolve_file_path(candidate)

        # 3. Error si no existe
        if not resolved_target:
            search_locations = [str(Path(sp) / file_path) for sp in (self.search_paths or [])]
            search_locations.append(str(base_file.parent / file_path))

            error_msg = f'// ERROR: File not found: {file_path}\n'
            error_msg += f'// Searched in:\n'
            for loc in search_locations:
                error_msg += f'//   - {loc}\n'
            return error_msg

        try:
            with open(resolved_target, 'r', encoding='utf-8') as f:
                content = f.read()

            if '@embedia-include' in content:
                content = self.process_content(content, resolved_target)

            header = '' # f'\n// ======== BEGIN: {file_path} ========\n'
            footer = '' # f'\n// ======== END: {file_path} ========\n'

            return header + content + footer

        except Exception as e:
            return f'// ERROR reading {file_path}: {str(e)}\n'

    def _remove_notes_blocks(self, content: str) -> str:
        """Elimina bloques de comentarios que contengan @embedia-note"""
        import re
        # Patrón para bloques /* ... */ (no greedy)
        pattern = r'/\*.*?\*/'
        
        # Eliminar bloques que contengan @embedia-note
        result = []
        last_end = 0
        for match in re.finditer(pattern, content, re.DOTALL):
            if self.NOTES in match.group(0):
                # Encontrado bloque con @embedia-note, omitirlo
                result.append(content[last_end:match.start()])
                last_end = match.end()
            else:
                # Bloque normal, mantenerlo
                result.append(content[last_end:match.end()])
                last_end = match.end()
        result.append(content[last_end:])
        
        return ''.join(result)

    def _resolve_file_path(self, file_path: Path) -> Optional[Path]:
        """
        Resuelve la ruta real de un archivo, manejando .lnk si existe.

        Returns:
            Path resuelto o None si no existe
        """
        # Si el archivo existe directamente
        if file_path.exists():
            return file_path

        # Verificar si existe un archivo .lnk
        link_file = file_path.with_suffix(file_path.suffix + '.lnk')
        if link_file.exists():
            try:
                # Leer la ruta del archivo real
                with open(link_file, 'r', encoding='utf-8') as f:
                    target_path = f.read().strip()

                # Eliminar comentarios (líneas que empiezan con #)
                if '#' in target_path:
                    target_path = target_path.split('#')[0].strip()

                # Resolver ruta
                if os.path.isabs(target_path):
                    resolved = Path(target_path)
                else:
                    # Ruta relativa desde el directorio del link_file
                    resolved = link_file.parent / target_path
                    resolved = resolved.resolve()

                return resolved if resolved.exists() else None

            except Exception:
                return None

        return None

    # ===== MÉTODOS DE UTILIDAD =====

    def file_has_directives(self, file_path: Path) -> bool:
        """
        Verifica rápidamente si un archivo tiene directivas.

        Optimización: Solo lee las primeras líneas.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 50:  # Solo primeras 50 líneas
                        break
                    if '@embedia-include' in line:
                        return True
            return False
        except Exception:
            return False

    def content_has_directives(self, content: str) -> bool:
        """
        Verifica si el contenido tiene directivas.

        Optimización: Salta comentarios iniciales/licencia (~30 líneas),
        luego busca en todo el contenido restante.
        """
        lines = content.split('\n')

        start_search = 30  # Saltar primeras 30 líneas (comentarios, licencia, includes)
        if start_search > len(lines): # no saltar más allá del archivo
            start_search = 0

        for i in range(start_search, len(lines)):
            if '@embedia-include' in lines[i]:
                return True

        return False

    def clear_cache(self):
        """Limpia la cache de archivos procesados"""
        self._processed_files_cache.clear()

    def process_file_full(self, src_path, dst_path, inject_headers=None, update_defines=None):
        """
        Procesa archivo con directivas + otras transformaciones.

        Args:
            src_path: Ruta origen (ya resuelta, sin .lnk)
            dst_path: Ruta destino
            inject_headers: str para reemplazar {includes}
            update_defines: dict {nombre: valor}
        """
        # Leer archivo
        try:
            content = read_from_file(src_path)
        except:
            raise FileNotFoundError(f'File not found: {src_path}')

        # 1. Procesar directivas @embedia-include
        if self.content_has_directives(content):
            content = self.process_content(content, Path(src_path))

        # 2. Inyectar headers
        if inject_headers and '{includes}' in content:
            content = content.replace('{includes}', inject_headers)

        # 3. Actualizar defines
        if update_defines:
            for name, value in update_defines.items():
                content = self._update_define(content, name, value)

        # Guardar
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        save_to_file(dst_path, content)


    def _update_define(self, content, name, value):
        """Actualiza o agrega un #define"""
        import re

        # Buscar #define existente
        pattern = rf'^(\s*#define\s+{re.escape(name)}\s+).*$'
        replacement = rf'\g<1>{value}'

        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        # No hubo reemplazo, agregar nuevo #define después de los includes
        if count == 0:
            lines = content.split('\n')

            # Buscar última línea con #include
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#include'):
                    insert_pos = i + 1

            # Agregar línea en blanco si no existe
            if insert_pos > 0 and insert_pos < len(lines) and lines[insert_pos].strip():
                lines.insert(insert_pos, '')
                insert_pos += 1

            lines.insert(insert_pos, f'#define {name} {value}')
            new_content = '\n'.join(lines)

        return new_content
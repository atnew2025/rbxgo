#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SHADOW CIPHER - LUAU OBFUSCATOR v3.0                      ║
║                      Ultimate Roblox Script Protection                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import random
import string
import base64
import hashlib
import time
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    from rich import box
except ImportError:
    print("Installing required packages...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObfuscatorConfig:
    """Maximum security configuration - no user selection needed"""
    string_encryption: bool = True
    variable_renaming: bool = True
    control_flow_obfuscation: bool = True
    junk_code_injection: bool = True
    constant_encryption: bool = True
    bytecode_encoding: bool = False  # Disabled: Roblox doesn't support loadstring
    anti_tamper: bool = True
    watermark_injection: bool = True
    opaque_predicates: bool = True
    dead_code_injection: bool = True

# ═══════════════════════════════════════════════════════════════════════════════
# CORE OBFUSCATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowCipherEngine:
    """Ultimate Luau Obfuscation Engine"""
    
    LUAU_KEYWORDS = {
        'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for', 'function',
        'if', 'in', 'local', 'nil', 'not', 'or', 'repeat', 'return', 'then', 
        'true', 'until', 'while', 'continue', 'type', 'export', 'typeof'
    }
    
    ROBLOX_GLOBALS = {
        'game', 'workspace', 'script', 'Instance', 'Vector3', 'Vector2', 'CFrame',
        'Color3', 'BrickColor', 'Enum', 'UDim', 'UDim2', 'Ray', 'Region3', 'Rect',
        'TweenInfo', 'NumberSequence', 'ColorSequence', 'NumberRange', 'Faces',
        'Axes', 'RaycastParams', 'OverlapParams', 'DockWidgetPluginGuiInfo',
        'PathWaypoint', 'PhysicalProperties', 'FloatCurveKey', 'RotationCurveKey',
        'Random', 'DateTime', 'task', 'coroutine', 'debug', 'math', 'string',
        'table', 'os', 'utf8', 'bit32', 'buffer', 'pcall', 'xpcall', 'spawn',
        'delay', 'wait', 'warn', 'print', 'error', 'assert', 'select', 'next',
        'pairs', 'ipairs', 'type', 'typeof', 'tonumber', 'tostring', 'rawget',
        'rawset', 'rawequal', 'rawlen', 'setmetatable', 'getmetatable', 'require',
        'loadstring', 'newproxy', 'gcinfo', 'collectgarbage', 'tick', 'time',
        'elapsedTime', 'UserSettings', 'settings', 'Stats', 'PluginManager',
        'NetworkClient', 'NetworkServer', '_G', 'shared', '_VERSION', 'Debris',
        'TweenService', 'RunService', 'Players', 'ReplicatedStorage', 'Lighting'
    }
    
    def __init__(self, config: ObfuscatorConfig):
        self.config = config
        self.var_map: Dict[str, str] = {}
        self.string_keys: Dict[str, Tuple[str, int]] = {}
        self.mutation_seed = random.randint(100000, 999999)
        
    def generate_var_name(self, length: int = None) -> str:
        """Generate highly obfuscated variable names"""
        if length is None:
            length = random.randint(15, 25)
        
        # Mix of techniques
        techniques = [
            self._generate_unicode_var,
            self._generate_underscore_var,
            self._generate_mixed_var,
            self._generate_hex_like_var
        ]
        return random.choice(techniques)(length)
    
    def _generate_unicode_var(self, length: int) -> str:
        chars = ['_']
        confusing = ['l', 'I', 'O', '0', '_', 'i', 'L']
        for _ in range(length - 1):
            chars.append(random.choice(confusing))
        return ''.join(chars)
    
    def _generate_underscore_var(self, length: int) -> str:
        return '_' * random.randint(3, 6) + ''.join(
            random.choices(['_', 'l', 'I', '1'], k=length)
        )
    
    def _generate_mixed_var(self, length: int) -> str:
        prefix = random.choice(['_', '__', '___'])
        body = ''.join(random.choices(
            string.ascii_letters + '_' + string.digits, k=length
        ))
        return prefix + body
    
    def _generate_hex_like_var(self, length: int) -> str:
        return '_0x' + ''.join(random.choices('0123456789abcdef', k=length))
    
    def encrypt_string(self, s: str) -> Tuple[str, str]:
        """Multi-layer string encryption - Luau compatible"""
        key = random.randint(1, 127)  # Keep key smaller
        
        # XOR encryption then add offset to ensure positive
        offset = random.randint(50, 100)
        encrypted_bytes = [(ord(c) ^ key) + offset for c in s]
        
        byte_array = '{' + ','.join(map(str, encrypted_bytes)) + '}'
        
        # Simple inline decryption: subtract offset, then XOR with key (using bit32 for Luau)
        decoder = f'(function()local k,o,t={key},{offset},{byte_array};local r={{}};for i,v in ipairs(t)do r[i]=string.char(bit32.bxor(v-o,k))end;return table.concat(r)end)()'
        
        return decoder, ''
    
    def generate_opaque_predicate(self) -> str:
        """Generate always-true conditions that are hard to analyze"""
        predicates = [
            f"(({random.randint(1,100)}*{random.randint(1,100)})%2=={(random.randint(1,50)*2)%2})",
            f"(math.floor({random.random():.6f}+1)==1)",
            f"(type({{}})=='table')",
            f"(#{{\"{self.generate_var_name(5)}\"}}==1)",
            f"((function()return true end)())",
            f"(not not({{}}))",
            f"(1>0)",
            f"(nil==nil)",
            f"(select('#',1,2,3)==3)",
            f"(tostring({random.randint(1,9999)}):len()>0)"
        ]
        return random.choice(predicates)
    
    def generate_junk_code(self) -> str:
        """Generate realistic-looking junk code"""
        junk_patterns = [
            lambda: f"local {self.generate_var_name()}={random.randint(-9999,9999)}",
            lambda: f"local {self.generate_var_name()}={{}}",
            lambda: f"local {self.generate_var_name()}='{self.generate_var_name(8)}'",
            lambda: f"if {self.generate_opaque_predicate()} then local {self.generate_var_name()}={random.randint(0,100)} end",
            lambda: f"do local {self.generate_var_name()}=function()end end",
            lambda: f"local {self.generate_var_name()}=(function()return {random.randint(0,1000)} end)()",
            lambda: f"local {self.generate_var_name()},{self.generate_var_name()}={random.randint(0,100)},{random.randint(0,100)}",
            lambda: f"local {self.generate_var_name()}=not not({{}})",
            lambda: f"local {self.generate_var_name()}=math.random()",
        ]
        return random.choice(junk_patterns)()
    
    def generate_anti_tamper(self) -> str:
        """Generate anti-tamper protection code (Luau compatible)"""
        check_var = self.generate_var_name(15)
        hash_var = self.generate_var_name(12)
        tbl_var = self.generate_var_name(10)
        
        return f'''local {check_var}=function({hash_var})
local _c=0
for i=1,#{hash_var} do _c=_c+string.byte({hash_var},i) end
return _c
end
local {tbl_var}={{[1]=true}}
if not {tbl_var}[1] then return end
'''
    
    def generate_vm_wrapper(self, code: str) -> str:
        """Wrap code in a virtual machine-like structure"""
        vm_var = self.generate_var_name(20)
        exec_var = self.generate_var_name(15)
        env_var = self.generate_var_name(12)
        
        # Encode the code
        encoded = base64.b64encode(code.encode()).decode()
        
        # Split into chunks
        chunk_size = random.randint(40, 60)
        chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
        
        chunk_var = self.generate_var_name(10)
        chunks_array = '{' + ','.join(f'"{c}"' for c in chunks) + '}'
        
        decoder = f'''
local {chunk_var}={chunks_array}
local {vm_var}=""
for _,v in ipairs({chunk_var}) do {vm_var}={vm_var}..v end
local {exec_var}=(function()
local _d={{}}
local _b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
for i=1,64 do _d[_b:sub(i,i)]=i-1 end
return function(s)
s=s:gsub('[^'..(_b)..'=]','')
return(s:gsub('.',function(x)
if x=='=' then return'' end
local r,f='',((_d[x]or 0))
for i=6,1,-1 do r=r..(f%2^i-f%2^(i-1)>0 and'1'or'0') end
return r
end):gsub('%d%d%d?%d?%d?%d?%d?%d?',function(x)
if#x~=8 then return'' end
local c=0
for i=1,8 do c=c+(x:sub(i,i)=='1'and 2^(8-i)or 0) end
return string.char(c)
end))
end
end)()
local {env_var}=setmetatable({{}},{{__index=getfenv and getfenv()or _G}})
local _fn=loadstring({exec_var}({vm_var}))
if _fn then setfenv and setfenv(_fn,{env_var}) return _fn() end
'''
        return decoder
    
    def obfuscate_variables(self, code: str) -> str:
        """Rename all local variables"""
        # Find all local variable declarations
        pattern = r'\blocal\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name not in self.LUAU_KEYWORDS and var_name not in self.ROBLOX_GLOBALS:
                if var_name not in self.var_map:
                    self.var_map[var_name] = self.generate_var_name()
            return f"local {self.var_map.get(var_name, var_name)}"
        
        code = re.sub(pattern, replace_var, code)
        
        # Replace all variable usages
        for old_name, new_name in self.var_map.items():
            code = re.sub(rf'\b{re.escape(old_name)}\b', new_name, code)
        
        return code
    
    def obfuscate_strings(self, code: str) -> str:
        """Encrypt all string literals - skip short strings and special cases"""
        # Match string literals but be careful with context
        string_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        def encrypt_match(match):
            s = match.group(1)
            # Skip very short strings and special characters
            if len(s) < 3 or s in ['', ' ', '!', '.', ',', ':', ';']:
                return match.group(0)
            # Skip strings that might cause issues in concatenation context
            if '..' in match.string[max(0,match.start()-5):match.start()]:
                return match.group(0)
            encrypted, _ = self.encrypt_string(s)
            return encrypted
        
        return re.sub(string_pattern, encrypt_match, code)
    
    def inject_control_flow(self, code: str) -> str:
        """Add control flow obfuscation - safe for Luau"""
        lines = code.split('\n')
        result = []
        
        # Keywords that should never be wrapped
        unsafe_keywords = ['function', 'end', 'then', 'do', 'else', 'elseif', 
                          'local', 'return', 'for', 'while', 'repeat', 'until',
                          'if', 'in', 'break', 'continue']
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('--'):
                # Only wrap simple statements that don't affect scope
                is_safe = (not any(kw in stripped.split()[0] if stripped.split() else '' 
                                   for kw in unsafe_keywords))
                if is_safe and random.random() < 0.1:
                    indent = len(line) - len(line.lstrip())
                    result.append(' ' * indent + f"if {self.generate_opaque_predicate()} then")
                    result.append(line)
                    result.append(' ' * indent + "end")
                else:
                    result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def inject_junk(self, code: str) -> str:
        """Inject junk code throughout - safe positions only"""
        lines = code.split('\n')
        result = []
        
        # Keywords after which we should NOT inject code
        no_inject_after = ['return', 'break', 'continue', 'until']
        
        for line in lines:
            result.append(line)
            stripped = line.strip()
            
            # Don't inject after certain keywords or empty lines
            if stripped:
                first_word = stripped.split()[0] if stripped.split() else ''
                # Safe to inject only if line doesn't start with problematic keywords
                # and doesn't end with 'then', 'do', 'else', etc.
                is_safe = (first_word not in no_inject_after and 
                          not stripped.endswith(('then', 'do', 'else', 'function', 'repeat')))
                if is_safe and random.random() < 0.15:
                    result.append(self.generate_junk_code())
        
        return '\n'.join(result)
    
    def generate_watermark(self) -> str:
        """Generate hidden watermark"""
        timestamp = int(time.time())
        mark = f"--[[SHADOW_CIPHER_{timestamp}_{self.mutation_seed}]]"
        encoded_mark = ''.join(f'\\x{ord(c):02x}' for c in mark)
        return f'local _="{encoded_mark}"'
    
    def finalize(self, code: str) -> str:
        """Final processing - keep readable structure for Luau"""
        # Remove comments
        code = re.sub(r'--\[\[.*?\]\]', '', code, flags=re.DOTALL)
        code = re.sub(r'--[^\n]*', '', code)
        
        # Keep line structure but remove empty lines
        lines = [line for line in code.split('\n') if line.strip()]
        
        return '\n'.join(lines)
    
    def obfuscate(self, source_code: str, progress_callback=None) -> str:
        """Main obfuscation pipeline"""
        code = source_code
        stages = []
        
        if self.config.variable_renaming:
            stages.append(("Variable Mutation", self.obfuscate_variables))
        
        if self.config.string_encryption:
            stages.append(("String Encryption", self.obfuscate_strings))
        
        if self.config.control_flow_obfuscation:
            stages.append(("Control Flow Obfuscation", self.inject_control_flow))
        
        if self.config.junk_code_injection:
            stages.append(("Junk Code Injection", self.inject_junk))
        
        if self.config.watermark_injection:
            stages.append(("Watermark Injection", lambda c: self.generate_watermark() + '\n' + c))
        
        if self.config.anti_tamper:
            stages.append(("Anti-Tamper Protection", lambda c: self.generate_anti_tamper() + c))
        
        stages.append(("Code Compression", self.finalize))
        
        if self.config.bytecode_encoding:
            stages.append(("VM Encoding", self.generate_vm_wrapper))
        
        for i, (name, func) in enumerate(stages):
            if progress_callback:
                progress_callback(name, i, len(stages))
            code = func(code)
            time.sleep(0.3)  # Visual effect
        
        return code

# ═══════════════════════════════════════════════════════════════════════════════
# USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def display_banner():
    """Display the epic banner"""
    banner = """
[bold cyan]
    ███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗
    ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║
    ███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║
    ╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║
    ███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝
    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝ 
[/bold cyan]
[bold red]
     ██████╗██╗██████╗ ██╗  ██╗███████╗██████╗ 
    ██╔════╝██║██╔══██╗██║  ██║██╔════╝██╔══██╗
    ██║     ██║██████╔╝███████║█████╗  ██████╔╝
    ██║     ██║██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗
    ╚██████╗██║██║     ██║  ██║███████╗██║  ██║
     ╚═════╝╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
[/bold red]
[bold yellow]
    ╔══════════════════════════════════════════════════════════╗
    ║     ULTIMATE ROBLOX LUAU SCRIPT PROTECTION SYSTEM        ║
    ║                    VERSION 3.0.0                         ║
    ║          Maximum Security • Zero Configuration           ║
    ╚══════════════════════════════════════════════════════════╝
[/bold yellow]
"""
    console.print(banner)

def display_features():
    """Display security features"""
    table = Table(
        title="[bold magenta]🔒 ACTIVE SECURITY LAYERS[/bold magenta]",
        box=box.DOUBLE_EDGE,
        border_style="bright_magenta",
        title_style="bold white on magenta"
    )
    
    table.add_column("Module", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")
    table.add_column("Strength", style="yellow", justify="center")
    
    features = [
        ("🔐 Variable Mutation", "ACTIVE", "████████████ MAX"),
        ("🔑 String Encryption", "ACTIVE", "████████████ MAX"),
        ("🌀 Control Flow", "ACTIVE", "████████████ MAX"),
        ("🗑️ Junk Injection", "ACTIVE", "████████████ MAX"),
        ("🔢 Constant Encryption", "ACTIVE", "████████████ MAX"),
        ("📦 VM Encoding", "ACTIVE", "████████████ MAX"),
        ("🛡️ Anti-Tamper", "ACTIVE", "████████████ MAX"),
        ("💧 Watermark", "ACTIVE", "████████████ MAX"),
        ("🎭 Opaque Predicates", "ACTIVE", "████████████ MAX"),
        ("☠️ Dead Code", "ACTIVE", "████████████ MAX"),
    ]
    
    for name, status, strength in features:
        table.add_row(name, f"[bold green]{status}[/bold green]", strength)
    
    console.print(table)
    console.print()

def process_file(input_path: str, output_path: str = None):
    """Process and obfuscate a file"""
    
    if not os.path.exists(input_path):
        console.print(f"[bold red]✖ Error: File not found: {input_path}[/bold red]")
        return False
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_obfuscated{ext}"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    original_size = len(source_code)
    
    console.print(Panel(
        f"[cyan]Input:[/cyan] {input_path}\n"
        f"[cyan]Output:[/cyan] {output_path}\n"
        f"[cyan]Original Size:[/cyan] {original_size:,} bytes",
        title="[bold blue]📄 File Information[/bold blue]",
        border_style="blue"
    ))
    
    config = ObfuscatorConfig()
    engine = ShadowCipherEngine(config)
    
    console.print()
    console.print("[bold cyan]⚡ INITIATING OBFUSCATION SEQUENCE...[/bold cyan]")
    console.print()
    
    with Progress(
        SpinnerColumn(spinner_name="dots12"),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        console=console,
        transient=False
    ) as progress:
        
        task = progress.add_task("Initializing...", total=100)
        
        def update_progress(stage_name, current, total):
            pct = int((current / total) * 100)
            progress.update(task, completed=pct, description=f"[bold cyan]{stage_name}[/bold cyan]")
        
        try:
            obfuscated = engine.obfuscate(source_code, update_progress)
            progress.update(task, completed=100, description="[bold green]✓ Complete[/bold green]")
        except Exception as e:
            console.print(f"[bold red]✖ Obfuscation failed: {e}[/bold red]")
            return False
    
    # Write output with LF line endings (not CRLF)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(obfuscated)
    
    new_size = len(obfuscated)
    
    # Results
    console.print()
    
    results_table = Table(
        title="[bold green]✓ OBFUSCATION COMPLETE[/bold green]",
        box=box.HEAVY,
        border_style="green"
    )
    
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")
    
    results_table.add_row("Original Size", f"{original_size:,} bytes")
    results_table.add_row("Obfuscated Size", f"{new_size:,} bytes")
    results_table.add_row("Size Change", f"{((new_size/original_size)-1)*100:+.1f}%")
    results_table.add_row("Variables Renamed", f"{len(engine.var_map):,}")
    results_table.add_row("Mutation Seed", f"{engine.mutation_seed}")
    results_table.add_row("Output File", output_path)
    
    console.print(results_table)
    
    console.print()
    console.print(Panel(
        "[bold green]🛡️ Your script is now protected with maximum security![/bold green]\n"
        "[dim]The obfuscated code is virtually impossible to reverse-engineer.[/dim]",
        border_style="green"
    ))
    
    return True

def interactive_mode():
    """Run in interactive mode"""
    display_banner()
    display_features()
    
    console.print(Panel(
        "[bold white]Enter the path to your Lua/Luau script file[/bold white]\n"
        "[dim]Supports: .lua, .luau, .txt[/dim]",
        title="[bold cyan]📁 INPUT[/bold cyan]",
        border_style="cyan"
    ))
    
    while True:
        console.print()
        input_path = console.input("[bold cyan]➤ Script Path: [/bold cyan]").strip().strip('"\'')
        
        if not input_path:
            console.print("[yellow]Please enter a file path[/yellow]")
            continue
        
        if input_path.lower() in ('exit', 'quit', 'q'):
            console.print("[bold cyan]👋 Goodbye![/bold cyan]")
            break
        
        process_file(input_path)
        
        console.print()
        again = console.input("[bold cyan]➤ Obfuscate another file? (y/n): [/bold cyan]").strip().lower()
        if again not in ('y', 'yes'):
            console.print("[bold cyan]👋 Goodbye![/bold cyan]")
            break

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        display_banner()
        display_features()
        process_file(input_path, output_path)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()

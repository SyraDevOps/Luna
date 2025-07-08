def clean_model_output(text):
    """
    Limpa a saída do modelo, eliminando problemas de formatação como
    palavras espalhadas verticalmente e espaços em excesso.
    """
    import re
    
    # Preservar alguns padrões específicos que podem ser legítimos
    placeholder_map = {}
    
    # Preservar URLs
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, text)
    for i, url in enumerate(urls):
        placeholder = f"__URL_PLACEHOLDER_{i}__"
        text = text.replace(url, placeholder)
        placeholder_map[placeholder] = url
        
    # Preservar blocos de código
    code_blocks = []
    code_pattern = r'```[\s\S]*?```'
    matches = re.findall(code_pattern, text)
    for i, block in enumerate(matches):
        placeholder = f"__CODE_PLACEHOLDER_{i}__"
        text = text.replace(block, placeholder)
        placeholder_map[placeholder] = block
    
    # Etapa 1: Consolidar linhas com palavras isoladas
    lines = text.split('\n')
    consolidated_lines = []
    buffer = ""
    
    for line in lines:
        stripped = line.strip()
        
        # Pular linhas vazias consecutivas
        if not stripped:
            if buffer:  # Se temos algo no buffer
                consolidated_lines.append(buffer)
                buffer = ""
            consolidated_lines.append("")  # Manter uma quebra de linha
            continue
            
        # Identificar linhas que são apenas palavras soltas ou fragmentos
        if len(stripped.split()) <= 3 and len(stripped) < 20:
            if buffer and not buffer.endswith(" "):
                buffer += " "
            buffer += stripped
        else:
            # Se é uma linha completa
            if buffer:
                consolidated_lines.append(buffer)
                buffer = ""
            consolidated_lines.append(stripped)
    
    # Não esquecer do último buffer
    if buffer:
        consolidated_lines.append(buffer)
    
    # Etapa 2: Reconectar linhas fragmentadas
    result = []
    i = 0
    while i < len(consolidated_lines):
        line = consolidated_lines[i]
        
        # Se linha atual não termina com pontuação e próxima linha existe e começa com minúscula
        if (i < len(consolidated_lines) - 1 and line and 
            not line.endswith(('.', '!', '?', ':', ';', '"', "'")) and 
            consolidated_lines[i+1] and consolidated_lines[i+1][0].islower()):
            line += " " + consolidated_lines[i+1]
            i += 2  # Pular a próxima linha que já foi incorporada
        else:
            i += 1
            
        result.append(line)
    
    # Juntar tudo de volta com quebras de linha apropriadas
    clean_text = '\n'.join(result)
    
    # Remover múltiplas quebras de linha
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    # Remover espaços duplos
    clean_text = re.sub(r' {2,}', ' ', clean_text)
    
    # Restaurar placeholders
    for placeholder, original in placeholder_map.items():
        clean_text = clean_text.replace(placeholder, original)
        
    return clean_text
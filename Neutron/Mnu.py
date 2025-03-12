from Neutron import brain

def listar_modelos():
    """Lista todos os modelos disponíveis (arquivos .pt)"""
    modelos = set()
    for file in brain.os.listdir():
        if file.endswith('.pt'):
            modelos.add(file[:-3])  # Remove a extensão .pt
    return modelos

def main():
    print("=== Sistema de Aprendizado com Contexto ===")
    print("Comandos especiais:")
    print("/contexto [nome] - Define um contexto")
    print("/csv [arquivo] - Treina com CSV")
    print("/compromisso [desc data hora] - Agenda compromisso")
    print("/trocar [modelo] - Troca de modelo")
    print("/treinar [épocas] - Treina com todos os neurônios")
    print("/sair - Encerra o programa\n")
    
    modelos = listar_modelos()
    if modelos:
        print("Modelos disponíveis:")
        for modelo in modelos:
            print(f" - {modelo}")
    else:
        print("Nenhum modelo encontrado. Um novo será criado.")
    
    modelo = input("Nome do modelo: ").strip() or "default"
    cerebro = brain.Cerebro(modelo)
    
    while True:
        entrada = input("\nVocê: ").strip()
        
        if entrada.lower() == '/sair':
            break
            
        elif entrada.startswith('/contexto'):
            novo_contexto = entrada.split(' ', 1)[-1]
            cerebro.mudar_contexto(novo_contexto)
            continue
            
        elif entrada.startswith('/csv'):
            try:
                arquivo = entrada.split(' ', 1)[-1]
                epochs = int(input("Número de épocas para treinar: "))
                cerebro.treinar_com_csv(arquivo, epochs)
            except ValueError:
                print("Número de épocas inválido!")
            continue
            
        elif entrada.startswith('/compromisso'):
            try:
                _, desc, data, hora = entrada.split(' ', 3)
                cerebro.adicionar_compromisso(desc, data, hora)
            except:
                print("Formato inválido! Use: /compromisso [descrição] [data] [hora]")
            continue
        
        elif entrada.startswith('/trocar'):
            novo_modelo = entrada.split(' ', 1)[-1]
            cerebro.fechar()
            cerebro = brain.Cerebro(novo_modelo)
            continue
            
        elif entrada.startswith('/treinar'):
            try:
                epochs = int(entrada.split(' ', 1)[-1])
                cerebro.treinar_modelo(epochs)
            except ValueError:
                print("Número de épocas inválido!")
            continue
        
        resposta = cerebro.processar_pergunta(entrada)
        print(f"Sistema: {resposta}")
        
        if "não sei" in resposta.lower():
            nova_resposta = input("Como devo responder? ")
            novo_contexto = input("Qual contexto? (deixe em branco para nenhum) ") or None
            cerebro.adicionar_neuronio(entrada, nova_resposta, novo_contexto)
    
    cerebro.fechar()
    print("Sistema encerrado!")

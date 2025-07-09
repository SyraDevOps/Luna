# Segurança do Projeto Luna

Obrigado por seu interesse em manter o Luna seguro! Este documento descreve como relatar vulnerabilidades e nossas diretrizes de segurança.

## 📣 Relatando Vulnerabilidades

Se você encontrar qualquer falha de segurança ou comportamento suspeito relacionado ao Luna, por favor, **não abra uma issue pública no GitHub**. Em vez disso, entre em contato diretamente:

- **Email de segurança:** ContatoSyra@outlook.com
- **Prazo de resposta:** até 72h úteis
- **Prazo de correção estimado:** até 14 dias após verificação

Forneça o máximo de detalhes possível:
- Versão do Luna
- Etapas para reproduzir
- Logs relevantes (evite dados pessoais)
- Sistema operacional e ambiente (GPU, Python, etc.)

## 🔐 Escopo de Segurança

O projeto considera como prioritárias vulnerabilidades relacionadas a:

- Execução remota de código (RCE)
- Injeção de comandos ou código malicioso via entrada do usuário
- Vazamento de informações sensíveis (e.g., arquivos, logs, feedbacks)
- Quebra do isolamento entre modos offline e online
- Persistência ou exfiltração não autorizada de dados
- Escalada de privilégios no ambiente local

## 🧪 Ambiente Seguro

O Luna é projetado para funcionar **localmente e offline** por padrão. Para segurança:

- **Evite expor portas de API na internet** sem proteção.
- Use em ambientes controlados e com permissões restritas.
- Desabilite o modo de aprendizado contínuo automático se estiver lidando com dados sensíveis.

## 🔄 Política de Atualização de Segurança

Correções de segurança são tratadas com **prioridade máxima** e liberadas em versões patch (ex: 2.6.4 → 2.6.5).

As atualizações serão descritas em detalhes no `CHANGELOG.md`.

## 📄 Licenciamento e Restrições

O Luna pode incluir dependências de terceiros. Siga as licenças correspondentes.

O uso do modelo ou do framework para atividades maliciosas, scraping abusivo, engenharia social automatizada ou violação de leis locais é estritamente proibido.

---

*Última atualização: 08/07/2025*

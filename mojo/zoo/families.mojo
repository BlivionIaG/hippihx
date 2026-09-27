# Family bind hooks for the Mojo/MAX authoring center.
#
# Host mirror: hippihx/_lib/families.py. These are views, not new V1 ids.
# hybrid.heap is caller-owned: plan sizes it, serve zeros it once, bind
# only views. attention.kda_scan and attention.dsa_nope are not hooks.
#
# Not a kernel body. Not a fatbin. Not dest-ready.

alias HOOK_QWEN_QSA = "qwen.qsa"
alias HOOK_QWEN_QSA_OP = "attention.qsa_indexer"
alias HOOK_QWEN_QSA_TENSORS = "groups"
alias HOOK_QWEN_QSA_DTYPES = "unpinned"

alias HOOK_QWEN_GDN = "qwen.gdn"
alias HOOK_QWEN_GDN_OP = "attention.gdn_scan"
alias HOOK_QWEN_GDN_TENSORS = "mixed_qkv,a,b,out,state"
alias HOOK_QWEN_GDN_DTYPES = "fp16,fp16,fp16,fp16,fp16|fp32"

alias HOOK_QWEN_PLE = "qwen.ple"
alias HOOK_QWEN_PLE_OP = "sequence.causal_conv"
alias HOOK_QWEN_PLE_TENSORS = "input,out,state"
alias HOOK_QWEN_PLE_DTYPES = "fp16|bf16,fp16|bf16,fp16|bf16"

alias HOOK_MOE_ROUTED = "moe.routed"
alias HOOK_MOE_ROUTED_OP = "moe.routed"
alias HOOK_MOE_ROUTED_TENSORS = "gate,up,down"
alias HOOK_MOE_ROUTED_DTYPES = "fp16,fp16,fp16"

alias HOOK_MOE_LEFTOVER_BF16 = "moe.leftover_bf16"
alias HOOK_MOE_LEFTOVER_BF16_OP = "moe.leftover_bf16"
alias HOOK_MOE_LEFTOVER_BF16_TENSORS = "dense"
alias HOOK_MOE_LEFTOVER_BF16_DTYPES = "bf16"

alias HOOK_HYBRID_HEAP = "hybrid.heap"
alias HOOK_HYBRID_HEAP_OPS = "attention.gdn_scan,sequence.causal_conv"
alias HOOK_HYBRID_HEAP_TENSORS = "heap"
alias HOOK_HYBRID_HEAP_DTYPES = "fp16|fp32"
alias HOOK_HYBRID_HEAP_CALLER_OWNED = True

alias LEFT_WITHOUT_NEW_BRIEF = "attention.kda_scan,attention.dsa_nope"

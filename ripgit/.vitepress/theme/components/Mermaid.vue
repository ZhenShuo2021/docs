<template>
    <div v-html="svgRef"></div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps({
    id: String,
    code: String,
})

const svgRef = ref('')

const renderMermaid = async (id: string, code: string) => {
    const mermaid = await import('mermaid')
    mermaid.default.initialize({ startOnLoad: false })
    const { svg } = await mermaid.default.render(id, code)
    return svg
}

onMounted(async () => {
    svgRef.value = await renderMermaid(props.id, decodeURIComponent(props.code))
})
</script>

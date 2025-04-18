<script setup lang="ts">
import { useData } from 'vitepress'
import { computed } from 'vue'
const { page } = useData()

const date = computed(() => {
    const updated = page.value.lastUpdated
    return updated ? new Date(updated) : new Date('2000-01-01T00:00:00Z')
})
</script>

<template>
    <div class="LastUpdated">
        <p>更新时间: {{ date.toLocaleString() }}</p>
    </div>
</template>

<style>
.LastUpdated {
    color: var(--vp-c-text-2);
}
</style>

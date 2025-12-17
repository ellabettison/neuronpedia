import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import * as Tooltip from '@radix-ui/react-tooltip';
import { HelpCircle } from 'lucide-react';
import { useState } from 'react';

const MAX_TOPK_COS_SIM_FEATURES_TO_FETCH = 50;
const MAX_TOPK_TO_SHOW_INITIALLY = 10;
const MAX_COSSIM_FEATURE_DESC_LENGTH = 128;

type ResidChannel = {
  id: string;
  description: string;
  layer: number;
  index: number;
};

const exampleResidChannels: ResidChannel[] = [
  { id: 'layer5.res.5', description: 'dog toys', layer: 3, index: 5 },
  { id: 'layer2.res.63', description: 'colors', layer: 2, index: 63 },
  { id: 'layer8.res.82', description: 'speaking quietly', layer: 8, index: 82 },
  { id: 'layer11.res.6', description: 'coffee mugs', layer: 11, index: 6 },
];

const exampleConnectedNeurons = [
  {
    modelId: 'gemma-2-2b',
    layer: 12,
    index: '3425',
    current: false,
    resChannels: [exampleResidChannels[2], exampleResidChannels[3]],
  },
  {
    modelId: 'gemma-2-2b',
    layer: 16,
    index: '25',
    current: true,
    resChannels: [exampleResidChannels[0], exampleResidChannels[1], exampleResidChannels[2]],
  }, // current
  {
    modelId: 'gemma-2-2b',
    layer: 20,
    index: '253',
    current: false,
    resChannels: [exampleResidChannels[1], exampleResidChannels[3]],
  },
];

export default function ConnectedNeuronsPane({
  currentNeuron,
}: {
  currentNeuron: NeuronWithPartialRelations | undefined;
}) {
  const [hoveredNeuronIndex, setHoveredNeuronIndex] = useState<string | null>(null);
  const [hoveredChannelId, setHoveredChannelId] = useState<string | null>(null);

  return (
    // TODO: hide if not relevant model
    <div
      className={`mt-2 hidden flex-col gap-x-2 overflow-hidden rounded-lg border bg-white px-3 pb-4 pt-2 text-xs shadow transition-all sm:mt-3 ${
        true ? 'sm:flex' : 'sm:hidden'
      }`}
    >
      <div className="mb-1.5 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
        Connected Neurons
        <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <button type="button">
                <HelpCircle className="h-2.5 w-2.5" />
              </button>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                TODO: description
                <Tooltip.Arrow className="fill-slate-500" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        </Tooltip.Provider>
      </div>
      <div className="relative mx-8 mb-3 mt-1 min-h-96 flex-col">
        {exampleResidChannels.map((channel, arrayIndex) => {
          // Check if this channel is connected to the hovered neuron
          const isConnectedToHoveredNeuron = hoveredNeuronIndex
            ? exampleConnectedNeurons
                .find((n) => n.index === hoveredNeuronIndex)
                ?.resChannels.some((c) => c.id === channel.id)
            : false;

          // Check if this channel is connected to the current neuron
          const isConnectedToCurrentNeuron = exampleConnectedNeurons
            .find((n) => n.current)
            ?.resChannels.some((c) => c.id === channel.id);

          // Check if this channel is being hovered or connected to hovered neuron
          const isHighlighted = hoveredChannelId === channel.id || isConnectedToHoveredNeuron;

          const defaultBackground = isConnectedToCurrentNeuron
            ? 'linear-gradient(to bottom, transparent 0%, rgba(3, 105, 161, 0.3) 10%, rgba(3, 105, 161, 0.3) 90%, transparent 100%)'
            : 'linear-gradient(to bottom, transparent 0%, rgba(148, 163, 184, 0.3) 10%, rgba(148, 163, 184, 0.3) 90%, transparent 100%)';
          const hoverBackground = isConnectedToCurrentNeuron
            ? 'linear-gradient(to bottom, transparent 0%, #0369a1 10%, #0369a1 90%, transparent 100%)'
            : 'linear-gradient(to bottom, transparent 0%, #94a3b8 10%, #94a3b8 90%, transparent 100%)';

          return (
            <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <div
                    key={channel.id}
                    className={`absolute h-96 min-h-96 w-[5px] cursor-pointer rounded-sm`}
                    style={{
                      left: `${128 + arrayIndex * 24}px`,
                      background: isHighlighted ? hoverBackground : defaultBackground,
                      transition: 'background 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      setHoveredChannelId(channel.id);
                      e.currentTarget.style.background = hoverBackground;
                    }}
                    onMouseLeave={(e) => {
                      setHoveredChannelId(null);
                      e.currentTarget.style.background = isConnectedToHoveredNeuron
                        ? hoverBackground
                        : defaultBackground;
                    }}
                  />
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="rounded bg-slate-200 px-5 py-3 text-xs text-slate-600"
                    sideOffset={3}
                    side="left"
                  >
                    <div className="font-mono text-sm font-bold">{channel.id}</div>
                    <div className="mb-3 text-[9px] font-bold uppercase text-slate-500">Residual Stream Channel</div>
                    <div>{`"${channel.description}"`}</div>
                    <div>Layer {channel.layer}</div>
                    <div>Index {channel.index}</div>
                    <div>[More...]</div>
                    <Tooltip.Arrow className="fill-slate-200" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            </Tooltip.Provider>
          );
        })}
        {exampleConnectedNeurons.map((neuron, arrayIndex) => (
          <div key={neuron.index}>
            {neuron.resChannels.map((channel, channelArrayIndex) => {
              const channelIndex = exampleResidChannels.findIndex((c) => c.id === channel.id);
              const neuronLeft = 128 + exampleResidChannels.length * 28;
              const neuronTop = 64 + arrayIndex * 116;
              const channelLeft = 128 + channelIndex * 24;

              // Check if this path should be highlighted
              const isNeuronHovered = hoveredNeuronIndex === neuron.index;
              const isChannelHovered = hoveredChannelId === channel.id;
              const isHighlighted = isNeuronHovered || isChannelHovered;

              return (
                <svg
                  key={`line-${neuron.index}-${channel.id}`}
                  className="pointer-events-none absolute"
                  style={{
                    left: 2.5,
                    top: 0,
                    width: '100%',
                    height: '100%',
                    overflow: 'visible',
                  }}
                >
                  <path
                    d={`M ${channelLeft + 2.5} ${64 + arrayIndex * 116 + 40 * channelArrayIndex - 10} 
                        Q ${channelLeft + 2.5 + (neuronLeft + 10 - channelLeft - 2.5) * 0.6} ${64 + arrayIndex * 116 + 40 * channelArrayIndex - 10},
                          ${channelLeft + 2.5 + (neuronLeft + 10 - channelLeft - 2.5) * 0.7} ${(64 + arrayIndex * 116 + 40 * channelArrayIndex - 10 + neuronTop + 10) / 2}
                        Q ${channelLeft + 2.5 + (neuronLeft + 10 - channelLeft - 2.5) * 0.85} ${neuronTop + 10},
                          ${neuronLeft + 10} ${neuronTop + 10}`}
                    stroke={
                      isHighlighted
                        ? neuron.current
                          ? '#0369a1'
                          : '#94a3b8'
                        : neuron.current
                          ? 'rgba(3, 105, 161, 0.3)'
                          : 'rgba(148, 163, 184, 0.3)'
                    }
                    strokeWidth="2"
                    fill="none"
                    style={{ transition: 'stroke 0.3s ease' }}
                  />
                </svg>
              );
            })}
            {/* <div
              className="absolute h-5 w-5 cursor-pointer rounded-full border-4 border-sky-700 opacity-80 transition-all hover:opacity-100"
              style={{ left: `${72 + exampleResidChannels.length * 28}px`, top: `${30 + arrayIndex * 128}px` }}
            >
              {neuron.modelId} @ {neuron.layer} #{neuron.index}
            </div> */}
          </div>
        ))}

        {exampleConnectedNeurons.map((neuron, arrayIndex) => {
          // Check if this neuron is connected to the hovered channel
          const isConnectedToHoveredChannel = hoveredChannelId
            ? neuron.resChannels.some((c) => c.id === hoveredChannelId)
            : false;

          return (
            <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <div
                    key={neuron.index}
                    className={`absolute h-5 w-5 cursor-pointer rounded-full transition-all ${
                      neuron.current ? 'bg-sky-700' : 'bg-slate-400'
                    }`}
                    style={{
                      left: `${128 + 8 + exampleResidChannels.length * 28}px`,
                      top: `${64 + arrayIndex * 116}px`,
                      opacity: isConnectedToHoveredChannel ? 1 : 1,
                      transform: isConnectedToHoveredChannel ? 'scale(1.2)' : 'scale(1)',
                      transition: 'transform 0.3s ease, opacity 0.3s ease',
                    }}
                    onMouseEnter={() => setHoveredNeuronIndex(neuron.index)}
                    onMouseLeave={() => setHoveredNeuronIndex(null)}
                  >
                    {neuron.current ? <div className="ml-6 mt-0.5 text-[8px] font-bold text-sky-700">CURRENT</div> : ''}
                    {/* <div>
                    {neuron.modelId.toUpperCase()} @ {neuron.layer}
                  </div>
                  <div>{neuron.index}</div>
                  <div>{neuron.resChannels.map((channel) => channel.description).join(', ')}</div> */}
                  </div>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="rounded bg-slate-200 px-5 py-3 text-xs text-slate-600"
                    sideOffset={3}
                    side="right"
                  >
                    <div className="font-mono text-sm font-bold">
                      sparse-model @ {neuron.layer} #{neuron.index}
                    </div>
                    <div className="mb-3 text-[9px] font-bold uppercase text-slate-500">Connected Neuron</div>
                    <div className="mb-1">
                      <span className="font-semibold">Resid Stream Channels</span>
                    </div>
                    {neuron.resChannels.map((channel) => (
                      <div key={channel.id} className="mb-1">
                        • {channel.id}: &quot;{channel.description}&quot;
                      </div>
                    ))}
                    <Tooltip.Arrow className="fill-slate-200" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            </Tooltip.Provider>
          );
        })}
      </div>
    </div>
  );
}

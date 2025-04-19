import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Container, Heading, Text, Section, Card, Grid } from '../styles/componentStyles'; // Ensure Grid is imported

const AboutModelContainer = styled(Container)`
    max-width: 1000px;
`;

const ModelSection = styled(motion(Card))`
    margin-bottom: ${({ theme }) => theme.space.lg};
    padding: ${({ theme }) => theme.space.xl};
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    background: ${({ theme }) => theme.colors.backgroundAlt};
    position: relative;
    overflow: hidden;

    &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, ${({ theme }) => theme.colors.primary}, ${({ theme }) => theme.colors.secondary || theme.colors.primaryHover });
        transition: width 0.3s ease;
    }

    &:hover::after {
        width: 6px;
    }

    transition: all 0.3s ease;

    &:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
`;

const SectionTitle = styled(Heading)`
    font-size: ${({ theme }) => theme.fontSizes['2xl']};
    margin-bottom: ${({ theme }) => theme.space.md};
    color: ${({ theme }) => theme.colors.primary};
    display: inline-flex;
    align-items: center;
    position: relative;

    &::before {
        content: '';
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: ${({ theme }) => theme.colors.secondary || theme.colors.primaryHover};
        margin-right: 12px;
        display: inline-block;
    }
`;

const SubHeading = styled(motion(Heading))`
    font-size: ${({ theme }) => theme.fontSizes.lg};
    margin-top: ${({ theme }) => theme.space.lg};
    margin-bottom: ${({ theme }) => theme.space.sm};
    color: ${({ theme }) => theme.colors.textPrimary};
    position: relative;
    display: inline-block;
    transition: color 0.2s ease;

    &:hover {
        color: ${({ theme }) => theme.colors.secondary || theme.colors.primaryHover};
    }
`;

const Code = styled(motion.code)`
    background-color: ${({ theme }) => theme.colors.backgroundCode || '#2d2d2d'};
    color: ${({ theme }) => theme.colors.primaryHover};
    padding: 0.3em 0.5em;
    border-radius: ${({ theme }) => theme.borderRadius.sm};
    font-size: 0.95em;
    font-family: 'Fira Code', 'Courier New', Courier, monospace;
    transition: background-color 0.2s ease;
    display: inline-block;

    &:hover {
        background-color: ${({ theme }) => theme.colors.primaryHover}20;
    }
`;

const ListContainer = styled(motion.ul)`
    margin-top: ${({ theme }) => theme.space.md};
    margin-bottom: ${({ theme }) => theme.space.md};
    padding-left: 1.5em;
`;

const ListItem = styled(motion.li)`
    margin-bottom: ${({ theme }) => theme.space.sm};
    padding-left: 0.5em;
    position: relative;
    color: ${({ theme }) => theme.colors.textSecondary};

    &::marker {
        color: ${({ theme }) => theme.colors.secondary || theme.colors.primaryHover};
        font-weight: bold;
    }

    transition: color 0.2s ease;

    &:hover {
        color: ${({ theme }) => theme.colors.textPrimary};
    }
`;

const PageHeaderContainer = styled(motion.div)`
    text-align: center;
    margin-bottom: ${({ theme }) => theme.space.xl};
`;

const ModelVisualSection = styled(motion(Card))`
    margin-bottom: ${({ theme }) => theme.space.xl};
    padding: ${({ theme }) => theme.space.xl};
    border-radius: 20px;
    background: ${({ theme }) => theme.colors.background};
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    position: relative;
`;

const ArchitectureImage = styled(motion.div)`
    width: 100%;
    height: 200px;
    background: ${({ theme }) => `linear-gradient(135deg, ${theme.colors.primary}40, ${theme.colors.secondary || theme.colors.primaryHover}30)`};
    border-radius: 12px;
    margin: ${({ theme }) => theme.space.md} 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
    box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.2);
`;

const TabsContainer = styled.div`
    display: flex;
    justify-content: center;
    margin-bottom: ${({ theme }) => theme.space.lg};
    gap: ${({ theme }) => theme.space.sm};
`;

const Tab = styled(motion.button)`
    padding: ${({ theme }) => `${theme.space.sm} ${theme.space.md}`};
    background: ${({ isActive, theme }) =>
            isActive
                    ? `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary || theme.colors.primaryHover})`
                    : theme.colors.backgroundAlt};
    color: ${({ isActive }) => isActive ? '#ffffff' : ({ theme }) => theme.colors.textPrimary};
    border: none;
    border-radius: 30px;
    cursor: pointer;
    font-weight: 500;
    outline: none;
    min-width: 120px;
    box-shadow: ${({ isActive }) => isActive ? '0 4px 15px rgba(0, 0, 0, 0.15)' : 'none'};
`;

const TabContent = styled(motion.div)`
    width: 100%;
    margin-top: ${({ theme }) => theme.space.md};
    position: relative;
`;

// --- FIX: Define LayerName, LayerDescription, LayerIcon BEFORE ModelLayerItem ---
const LayerIcon = styled.div`
    width: 40px;
    height: 40px;
    flex-shrink: 0;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: ${({ theme }) => theme.space.md};
    background: ${({ theme }) => `linear-gradient(135deg, ${theme.colors.primary}50, ${theme.colors.secondary || theme.colors.primaryHover}50)`};
    color: ${({ theme }) => theme.colors.primary}; /* Initial icon number color */
    font-weight: bold;
    transition: color 0.2s ease, background 0.2s ease; /* Added transition */
`;

const LayerName = styled.h4`
    margin: 0 0 4px 0;
    font-size: ${({ theme }) => theme.fontSizes.md};
    color: ${({ theme }) => theme.colors.textPrimary}; /* Initial text color */
    transition: color 0.2s ease; /* Added transition */
`;

const LayerDescription = styled.p`
    margin: 0;
    font-size: ${({ theme }) => theme.fontSizes.sm};
    color: ${({ theme }) => theme.colors.textSecondary}; /* Initial text color */
    transition: color 0.2s ease; /* Added transition */
`;

// Now define ModelLayerItem using the components above
const ModelLayerItem = styled(motion.div)`
    padding: ${({ theme }) => theme.space.md};
    background: ${({ theme }) => theme.colors.backgroundAlt};
    border-radius: 12px;
    margin-bottom: ${({ theme }) => theme.space.sm};
    display: flex;
    align-items: center;
    cursor: default;
    border-left: 3px solid transparent;
    transition: background-color 0.2s ease, border-color 0.2s ease;

    &:hover {
        background: ${({ theme }) => theme.colors.textPrimary}; // Change background to white/light
        border-left-color: ${({ theme }) => theme.colors.secondary || theme.colors.primaryHover};

        /* Target child text/icon elements using the component names */
        ${LayerName}, ${LayerDescription} {
             color: ${({ theme }) => theme.colors.background}; /* Change text to dark */
        }
        ${LayerIcon} {
            color: ${({ theme }) => theme.colors.background}; /* Change icon number color too */
            background: ${({ theme }) => `linear-gradient(135deg, ${theme.colors.primary}30, ${theme.colors.secondary || theme.colors.primaryHover}30)`}; /* Optional: Slightly fade icon background */
        }
    }
`;
// --- End of FIX ---

const LayerContent = styled.div`
    flex: 1;
`;

const DatasetSection = styled(motion(Card))`
    margin-top: ${({ theme }) => theme.space.xl};
    padding: ${({ theme }) => theme.space.xl};
    border-radius: 20px;
    background: ${({ theme }) => theme.colors.backgroundAlt};
    overflow: hidden;
    position: relative;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);

    &::before {
        content: '';
        position: absolute;
        top: -50px;
        right: -50px;
        width: 150px;
        height: 150px;
        background: ${({ theme }) => `radial-gradient(circle, ${theme.colors.primary}15, transparent 70%)`};
        border-radius: 50%;
        opacity: 0.5;
        pointer-events: none;
    }

    transition: all 0.3s ease;

    &:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
`;


const AboutModelPage = () => {
    const [activeTab, setActiveTab] = useState('architecture');

    // Animation variants
    const pageVariants = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { duration: 0.6 } } };
    const containerStaggerVariants = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { delayChildren: 0.2, staggerChildren: 0.15 } } };
    const itemFadeUpVariants = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 100 } } };
    const tabContentVariants = {
        hidden: { opacity: 0, x: -10 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.4 } },
        exit: { opacity: 0, x: 10, transition: { duration: 0.2 } }
    };
    const tabVariants = { hover: { y: -3, transition: { duration: 0.2 } }, tap: { scale: 0.97 } };
    const codeVariants = { hover: { scale: 1.05, y: -1 }, tap: { scale: 0.95 } };
    const subHeadingVariants = { hover: { x: 5, color: ({ theme }) => theme.colors.secondary || theme.colors.primaryHover } }; // Use theme color
    const modelLayerItemVariants = {
        rest: { x: 0 },
        hover: { x: 5, transition: { type: 'spring', stiffness: 300, damping: 15 } }
    };


    // Model layer examples
    const modelLayers = [
        { id: 'conv1', name: 'Input & Convolution', description: 'Initial feature extraction (e.g., 7x7 Conv)' },
        { id: 'pool1', name: 'Pooling Layer', description: 'Downsampling feature maps (e.g., MaxPool)' },
        { id: 'resblock1', name: 'Residual Block(s)', description: 'Core feature learning blocks (multiple layers)' },
        { id: 'pool2', name: 'Final Pooling', description: 'Global Average Pooling for spatial reduction' },
        { id: 'fc', name: 'Fully Connected', description: 'Classification layer mapping features to outputs' }
    ];

    return (
        <Section as={motion.section} variants={pageVariants} initial="hidden" animate="visible">
            <AboutModelContainer>
                <PageHeaderContainer variants={itemFadeUpVariants}>
                    <Heading size="3xl" mb="md" align="center">
                        About Our AI Model & Project
                    </Heading>
                    <Text size="lg" color="textSecondary" align="center" mb="lg">
                        Discover the technology and process behind the X-Ray Diagnostic Assistant.
                    </Text>
                </PageHeaderContainer>

                {/* Interactive Model Visualization Section */}
                <ModelVisualSection variants={itemFadeUpVariants}>
                    <SectionTitle>Interactive Model Explorer</SectionTitle>
                    <TabsContainer>
                        <Tab isActive={activeTab === 'architecture'} onClick={() => setActiveTab('architecture')} variants={tabVariants} whileHover="hover" whileTap="tap"> Architecture </Tab>
                        <Tab isActive={activeTab === 'preprocessing'} onClick={() => setActiveTab('preprocessing')} variants={tabVariants} whileHover="hover" whileTap="tap"> Preprocessing </Tab>
                        <Tab isActive={activeTab === 'output'} onClick={() => setActiveTab('output')} variants={tabVariants} whileHover="hover" whileTap="tap"> Output </Tab>
                    </TabsContainer>

                    <AnimatePresence mode="wait">
                        <TabContent key={activeTab} variants={tabContentVariants} initial="hidden" animate="visible" exit="exit">
                            {activeTab === 'architecture' && (
                                <motion.div variants={containerStaggerVariants}>
                                    <Text as={motion.p} variants={itemFadeUpVariants}>
                                        Our core classification model utilizes the <Code variants={codeVariants} whileHover="hover" whileTap="tap">ResNet18</Code> architecture, known for balancing performance and efficiency in image tasks.
                                    </Text>
                                    <ArchitectureImage whileHover={{ scale: 1.02 }} transition={{ duration: 0.3 }}>
                                        Simplified ResNet18 Flow
                                    </ArchitectureImage>
                                    <Text as={motion.p} variants={itemFadeUpVariants}> Key layer types include:</Text>
                                    <motion.div variants={containerStaggerVariants}>
                                        {modelLayers.map((layer, index) => (
                                            <ModelLayerItem
                                                key={layer.id}
                                                variants={modelLayerItemVariants}
                                                whileHover="hover"
                                                transition={{ delay: index * 0.05 }}
                                            >
                                                <LayerIcon>{index + 1}</LayerIcon>
                                                <LayerContent> <LayerName>{layer.name}</LayerName> <LayerDescription>{layer.description}</LayerDescription> </LayerContent>
                                            </ModelLayerItem>
                                        ))}
                                    </motion.div>
                                </motion.div>
                            )}
                            {activeTab === 'preprocessing' && (
                                <motion.div variants={containerStaggerVariants}>
                                    <Text as={motion.p} variants={itemFadeUpVariants}> Input images undergo several steps:</Text>
                                    <ListContainer variants={containerStaggerVariants}>
                                        <ListItem variants={itemFadeUpVariants}>Image format conversion (to grayscale).</ListItem>
                                        <ListItem variants={itemFadeUpVariants}>Resizing to standard <Code variants={codeVariants} whileHover="hover" whileTap="tap">512x512</Code> pixels.</ListItem>
                                        <ListItem variants={itemFadeUpVariants}>Contrast enhancement using <Code variants={codeVariants} whileHover="hover" whileTap="tap">CLAHE</Code>. </ListItem>
                                        <ListItem variants={itemFadeUpVariants}>Normalization using custom dataset statistics (mean and std) from the training set.</ListItem>
                                        <ListItem variants={itemFadeUpVariants}>Conversion to <Code variants={codeVariants} whileHover="hover" whileTap="tap">.npy</Code> format for training.</ListItem>
                                        <ListItem variants={itemFadeUpVariants}>Window leveling applied with <Code variants={codeVariants} whileHover="hover" whileTap="tap">wl = 600</Code>, <Code variants={codeVariants} whileHover="hover" whileTap="tap">ww = 1500</Code>.</ListItem>
                                    </ListContainer>
                                    <Text as={motion.p} variants={itemFadeUpVariants}> This ensured higher consistency and improved our models robustness. </Text>
                                </motion.div>
                            )}
                            {activeTab === 'output' && (
                                <motion.div variants={containerStaggerVariants}>
                                    <Text as={motion.p} variants={itemFadeUpVariants}> The model outputs confidence scores for:</Text>
                                    <ListContainer variants={containerStaggerVariants}>
                                        <ListItem variants={itemFadeUpVariants}> <Code variants={codeVariants} whileHover="hover" whileTap="tap">NORMAL</Code>: No detected pathology. </ListItem>
                                        <ListItem variants={itemFadeUpVariants}> <Code variants={codeVariants} whileHover="hover" whileTap="tap">PNEUMONIA</Code>: Pneumonia patterns detected. </ListItem>
                                        <ListItem variants={itemFadeUpVariants}> <Code variants={codeVariants} whileHover="hover" whileTap="tap">TUBERCULOSIS</Code>: Tuberculosis signs detected. </ListItem>
                                        <ListItem variants={itemFadeUpVariants}> <Code variants={codeVariants} whileHover="hover" whileTap="tap">UNKNOWN</Code>: Uncertain or other pathology. </ListItem>
                                    </ListContainer>
                                    <Text as={motion.p} variants={itemFadeUpVariants}> The highest score is the primary prediction, but all scores are considered. </Text>
                                </motion.div>
                            )}
                        </TabContent>
                    </AnimatePresence>
                </ModelVisualSection>

                {/* --- FIX: Conditionally render the following sections --- */}
                {activeTab === 'architecture' && (
                    <motion.div variants={containerStaggerVariants} initial="hidden" animate="visible">
                        {/* Technical Details Section */}
                        <ModelSection variants={itemFadeUpVariants} whileHover={{ y: -5 }}>
                            <SectionTitle>Technical Details</SectionTitle>
                            {/* Training Process description - Subheading removed */}
                            <Text as={motion.p} variants={itemFadeUpVariants}>
                                The model was trained using a combination of public chest X-ray datasets, with strategic data augmentation
                                to improve generalization. We employed transfer learning starting with ImageNet weights to leverage
                                pre-trained feature extractors, then fine-tuned all layers using our specialized medical imaging dataset.
                            </Text>

                            {/* Performance Metrics Subheading and List */}
                            <SubHeading as="h3" variants={subHeadingVariants} whileHover="hover">Performance Metrics</SubHeading>
                            <Text as={motion.p} variants={itemFadeUpVariants}>
                                Our model achieves excellent performance metrics across all diagnostic categories:
                            </Text>
                            <ListContainer variants={containerStaggerVariants}>
                                <ListItem variants={itemFadeUpVariants}> Overall accuracy: <Code variants={codeVariants} whileHover="hover" whileTap="tap">99.42%</Code> on validation data </ListItem>
                                <ListItem variants={itemFadeUpVariants}> Sensitivity: <Code variants={codeVariants} whileHover="hover" whileTap="tap">98.61%</Code> for pneumonia, <Code variants={codeVariants} whileHover="hover" whileTap="tap">98.57%</Code> for tuberculosis </ListItem>
                                <ListItem variants={itemFadeUpVariants}> Specificity: <Code variants={codeVariants} whileHover="hover" whileTap="tap">99.42%</Code> for pneumonia, <Code variants={codeVariants} whileHover="hover" whileTap="tap">99.5%</Code> for tuberculosis </ListItem>
                                <ListItem variants={itemFadeUpVariants}> F1 Score: <Code variants={codeVariants} whileHover="hover" whileTap="tap">0.9942</Code> across all categories </ListItem>
                            </ListContainer>
                        </ModelSection>

                        {/* Dataset Section */}
                        <DatasetSection variants={itemFadeUpVariants} whileHover={{ y: -5 }}>
                            <SectionTitle>Training Data & Ethics</SectionTitle>
                            <Text as={motion.p} variants={itemFadeUpVariants}> Trained on around 10000 anonymized chest X-rays from public and clinical sources, ensuring category balance and attention to bias mitigation. </Text>
                            <SubHeading as="h3" variants={subHeadingVariants} >Data Privacy & Ethics</SubHeading>
                            <Text as={motion.p} variants={itemFadeUpVariants}> All training data was anonymized and processed in compliance with relevant privacy regulations. The model was developed with careful attention to bias mitigation, ensuring equitable performance across demographic groups and X-ray capture conditions. </Text>
                        </DatasetSection>
                    </motion.div>
                )} {/* --- End of conditional rendering block --- */}

            </AboutModelContainer>
        </Section>
    );
};

export default AboutModelPage;